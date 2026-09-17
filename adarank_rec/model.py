"""AdaRank-Rec model, independent of a particular Hugging Face decoder."""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .config import AdaRankConfig
from .modules import OrderedLowRankFusion, RankController


class AdaRankRec(nn.Module):
    def __init__(self, num_items: int, sid_table: Tensor, cfg: AdaRankConfig,
                 decoder: nn.Module | None = None):
        super().__init__()
        self.cfg, self.sid_table, self.decoder = cfg, sid_table.long(), decoder
        self.item_embedding = nn.Embedding(num_items + 1, cfg.dim, padding_idx=0)
        self.position_embedding = nn.Embedding(cfg.max_history, cfg.dim)
        self.visual_projection = nn.Linear(cfg.visual_dim, cfg.dim)
        self.text_projection = nn.Linear(cfg.text_dim, cfg.dim)
        self.query_tokens = nn.Parameter(torch.randn(cfg.num_queries, cfg.dim) * .02)
        layer = nn.TransformerEncoderLayer(cfg.dim, 8, 4 * cfg.dim, cfg.dropout, batch_first=True)
        self.modal_encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.fusion = OrderedLowRankFusion(cfg.dim, cfg.max_rank)
        self.context_adapter = nn.Sequential(nn.LayerNorm(cfg.dim), nn.Linear(cfg.dim, cfg.decoder_dim))
        self.controller = RankController(cfg.dim, len(cfg.ranks), cfg.controller_hidden)
        # Compact SID prediction head.
        self.sid_head = nn.Sequential(nn.Linear(cfg.decoder_dim, cfg.decoder_dim), nn.GELU(),
                                      nn.Linear(cfg.decoder_dim, 16 * cfg.codes_per_level))

    def encode(self, batch: dict) -> tuple[Tensor, Tensor, Tensor]:
        h = self.item_embedding(batch["history"])
        positions = torch.arange(h.shape[1], device=h.device)
        h = h + self.position_embedding(positions)[None]
        v, t = self.visual_projection(batch["visual"]), self.text_projection(batch["text"])
        q0 = self.query_tokens[None].expand(h.shape[0], -1, -1)
        query = self.modal_encoder(torch.cat([q0, v[:, None], t[:, None]], 1))[:, :self.cfg.num_queries]
        return h, query, batch["history_mask"]

    def logits_for_rank(self, batch: dict, rank: int, sid_targets: Tensor | None = None) -> Tensor:
        history, query, mask = self.encode(batch)
        z = self.fusion(query, history, mask, rank)
        context = self.context_adapter(z)
        if self.decoder is not None:
            if sid_targets is None and "target" in batch:
                sid_targets = self.sid_targets(batch["target"])
            return self.decoder(context, sid_targets)
        # Compact decoder path.
        return self.sid_head(context.mean(1)).view(-1, 4, 4 * self.cfg.codes_per_level)

    def controller_logits(self, batch: dict, budget: float | Tensor) -> Tensor:
        h, q, mask = self.encode(batch)
        return self.controller(h, q, mask, budget)

    def forward(self, batch: dict, budget: float | Tensor, rank: int | None = None):
        logits = self.controller_logits(batch, budget)
        chosen = self.cfg.ranks[logits.argmax(-1)[0].item()] if rank is None else rank
        return self.logits_for_rank(batch, chosen), logits

    def sid_targets(self, item_ids: Tensor) -> Tensor:
        return self.sid_table.to(item_ids.device)[item_ids]

    def recommendation_loss(self, logits: Tensor, target_items: Tensor) -> Tensor:
        target = self.sid_targets(target_items)
        return F.cross_entropy(logits.flatten(0, 1), target.flatten())

    @torch.no_grad()
    def score_candidates(self, batch: dict, candidates: Tensor, budget: float = .47) -> Tensor:
        """Teacher-force every 4-token SID and return [batch, candidates] scores.

        Requests are bucketed by the hard controller decision, so inactive Q/K
        prefixes are not evaluated for that request at serving time.
        """
        device = batch["history"].device
        candidates = candidates.to(device)
        routes = self.controller_logits(batch, budget).argmax(-1)
        scores = torch.empty(candidates.shape, device=device)
        for index, rank in enumerate(self.cfg.ranks):
            request_indices = (routes == index).nonzero(as_tuple=True)[0]
            if not len(request_indices):
                continue
            sub_batch = {key: value[request_indices] for key, value in batch.items()}
            items = candidates[request_indices].reshape(-1)
            if self.decoder is None:
                token_logits = self.logits_for_rank(sub_batch, rank)
                logp = token_logits.log_softmax(-1)
                sid = self.sid_targets(candidates[request_indices])
                values = logp[:, None].expand(-1, candidates.shape[1], -1, -1).gather(-1, sid[..., None]).squeeze(-1).sum(-1)
            else:
                # Autoregressive decoder input depends on the candidate prefix.
                repeated = {key: value.repeat_interleave(candidates.shape[1], 0) for key, value in sub_batch.items()}
                token_logits = self.logits_for_rank(repeated, rank, self.sid_targets(items))
                values = token_logits.log_softmax(-1).gather(-1, self.sid_targets(items)[..., None]).squeeze(-1).sum(-1)
                values = values.view(len(request_indices), -1)
            scores[request_indices] = values
        return scores

    @torch.no_grad()
    def sufficient_ranks(self, batch: dict, epsilon: float | None = None) -> Tensor:
        epsilon = self.cfg.epsilon if epsilon is None else epsilon
        target = batch["target"]
        losses = []
        for rank in self.cfg.ranks:
            l = F.cross_entropy(self.logits_for_rank(batch, rank).transpose(1, 2), self.sid_targets(target), reduction="none").sum(1)
            losses.append(l)
        values = torch.stack(losses, 1)
        gap = (values - values[:, -1:]).clamp_min(0)
        suffix = torch.flip(torch.cummax(torch.flip(gap, (1,)), 1).values, (1,))
        return (suffix <= epsilon).float().argmax(1)
