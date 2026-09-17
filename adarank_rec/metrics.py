from __future__ import annotations
import math
import torch


def ranking_metrics(scores: torch.Tensor, positive_index: int = 0, k: int = 10) -> dict:
    """scores [N, 21], with one positive candidate in each row."""
    order = scores.argsort(dim=-1, descending=True)
    ranks = (order == positive_index).nonzero()[:, 1] + 1
    hit = (ranks <= k).float()
    ndcg = hit / torch.log2(ranks.float() + 1)
    return {f"HR@{k}": hit.mean().item(), f"NDCG@{k}": ndcg.mean().item()}


def hard_cost(rank_indices: torch.Tensor, ranks=(32, 64, 128, 256), budget: float = .47) -> dict:
    selected = torch.tensor(ranks, device=rank_indices.device)[rank_indices]
    cost = selected.float().mean() / ranks[-1]
    return {"mean_rank": selected.float().mean().item(), "cost": cost.item(), "residual": (cost - budget).item()}
