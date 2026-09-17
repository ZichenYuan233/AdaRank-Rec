"""Ordered spectral fusion and budget-conditioned controller."""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class OrderedLowRankFusion(nn.Module):
    def __init__(self, dim: int, max_rank: int):
        super().__init__()
        self.uq = nn.Parameter(torch.empty(dim, max_rank))
        self.uk = nn.Parameter(torch.empty(dim, max_rank))
        self.raw_delta = nn.Parameter(torch.zeros(max_rank))
        nn.init.orthogonal_(self.uq)
        nn.init.orthogonal_(self.uk)

    def scales(self) -> Tensor:
        delta = F.softplus(self.raw_delta)
        return torch.flip(torch.cumsum(torch.flip(delta, (0,)), 0), (0,)) / delta.sum()

    def orthogonality_loss(self) -> Tensor:
        eye = torch.eye(self.uq.shape[1], device=self.uq.device, dtype=self.uq.dtype)
        return (self.uq.T @ self.uq - eye).square().sum() + (self.uk.T @ self.uk - eye).square().sum()

    def forward(self, query: Tensor, history: Tensor, mask: Tensor, rank: int) -> Tensor:
        if rank > self.uq.shape[1] or rank <= 0:
            raise ValueError(f"invalid rank {rank}")
        sigma = self.scales()[:rank]
        root = sigma.sqrt()
        qr = (query @ self.uq[:, :rank]) * root
        kr = (history @ self.uk[:, :rank]) * root
        logits = qr @ kr.transpose(-1, -2) / sigma.square().sum().sqrt().clamp_min(1e-8)
        logits = logits.masked_fill(~mask[:, None, :], torch.finfo(logits.dtype).min)
        attention = logits.softmax(-1)
        return attention @ history


class RankController(nn.Module):
    def __init__(self, dim: int, num_ranks: int, hidden: int):
        super().__init__()
        # normalized mean history + normalized mean query + dispersion/alignment/length + budget
        self.net = nn.Sequential(nn.Linear(2 * dim + 4, hidden), nn.GELU(), nn.Linear(hidden, num_ranks))
        self.h_ln, self.q_ln = nn.LayerNorm(dim), nn.LayerNorm(dim)

    def descriptors(self, history: Tensor, query: Tensor, mask: Tensor) -> Tensor:
        count = mask.sum(-1, keepdim=True).clamp_min(1)
        mean_h = (history * mask[..., None]).sum(1) / count
        norm_h = F.normalize(history, dim=-1)
        mean_norm_h = (norm_h * mask[..., None]).sum(1) / count
        dispersion = ((norm_h - mean_norm_h[:, None]).square().sum(-1) * mask).sum(1, keepdim=True) / count
        mean_q = query.mean(1)
        alignment = F.cosine_similarity(mean_h, mean_q, dim=-1).unsqueeze(-1)
        length = count.float().log1p()
        return torch.cat([self.h_ln(mean_h), self.q_ln(mean_q), dispersion, alignment, length], -1)

    def forward(self, history: Tensor, query: Tensor, mask: Tensor, budget: Tensor | float) -> Tensor:
        c = self.descriptors(history, query, mask)
        budget = torch.as_tensor(budget, dtype=c.dtype, device=c.device).reshape(-1, 1)
        if budget.numel() == 1: budget = budget.expand(c.shape[0], 1)
        return self.net(torch.cat([c, budget], -1))


def straight_through_gumbel(logits: Tensor, temperature: float) -> Tensor:
    return F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
