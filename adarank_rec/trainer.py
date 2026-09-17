"""Exact three-stage optimizer from the paper."""
from __future__ import annotations
import random
import torch
from torch import nn
import torch.nn.functional as F
from .modules import straight_through_gumbel


class AdaRankTrainer:
    def __init__(self, model, optimizer: torch.optim.Optimizer):
        self.model, self.optimizer = model, optimizer
        self.dual = {b: torch.tensor(0., device=next(model.parameters()).device) for b in model.cfg.budgets}

    def _move(self, batch):
        device = next(self.model.parameters()).device
        return {k: v.to(device) for k, v in batch.items()}

    def stage1_step(self, batch):
        batch = self._move(batch); ranks = (self.model.cfg.ranks[0], random.choice(self.model.cfg.ranks[1:-1]), self.model.cfg.ranks[-1])
        logits = [self.model.logits_for_rank(batch, r) for r in ranks]
        rec = sum(self.model.recommendation_loss(x, batch["target"]) for x in logits) / len(logits)
        teacher = logits[-1].detach().softmax(-1)
        consistency = sum(F.kl_div(x.log_softmax(-1), teacher, reduction="batchmean") for x in logits[:-1]) / 2
        loss = rec + self.model.cfg.orth_weight * self.model.fusion.orthogonality_loss() + self.model.cfg.consistency_weight * consistency
        self.optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.model.parameters(), self.model.cfg.grad_clip); self.optimizer.step()
        return {"loss": loss.item()}

    def stage2_step(self, batch, budget=.47):
        batch = self._move(batch)
        labels = self.model.sufficient_ranks(batch)
        loss = F.cross_entropy(self.model.controller_logits(batch, budget), labels)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        return {"loss": loss.item()}

    def stage3_step(self, batch, epoch: int, total_epochs: int):
        batch = self._move(batch); cfg = self.model.cfg; budget = random.choice(cfg.budgets)
        progress = epoch / max(total_epochs - 1, 1); temperature = 1.0 - .8 * progress
        logits = self.model.controller_logits(batch, budget)
        pi = logits.softmax(-1); route = straight_through_gumbel(logits, temperature)
        # Evaluate every branch then combine through the straight-through one-hot routing.
        branch_logits = torch.stack([self.model.logits_for_rank(batch, r) for r in cfg.ranks], 1)
        combined = (route[:, :, None, None] * branch_logits).sum(1)
        rec = self.model.recommendation_loss(combined, batch["target"])
        costs = torch.tensor(cfg.ranks, device=pi.device, dtype=pi.dtype) / cfg.max_rank
        expected_cost = (pi * costs).sum(-1).mean()
        labels = self.model.sufficient_ranks(batch)
        rank_loss = F.cross_entropy(logits, labels)
        teacher = branch_logits[:, -1].detach().softmax(-1)
        consistency = sum(F.kl_div(branch_logits[:, i].log_softmax(-1), teacher, reduction="batchmean")
                          for i in range(len(cfg.ranks)-1)) / (len(cfg.ranks)-1)
        mean_pi = pi.mean(0); entropy_loss = (torch.log(torch.tensor(len(cfg.ranks), device=pi.device)) + (mean_pi * mean_pi.clamp_min(1e-8).log()).sum()).square()
        with torch.no_grad(): self.dual[budget].clamp_min_(0).add_(cfg.dual_lr * (expected_cost.detach() - budget)).clamp_min_(0)
        loss = (rec + self.dual[budget] * (expected_cost - budget) + cfg.orth_weight * self.model.fusion.orthogonality_loss()
                + (1-progress) * cfg.consistency_weight * consistency + (1-progress) * rank_loss
                + (1-progress) * .01 * entropy_loss)
        self.optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip); self.optimizer.step()
        return {"loss": loss.item(), "cost": expected_cost.item(), "budget": budget}
