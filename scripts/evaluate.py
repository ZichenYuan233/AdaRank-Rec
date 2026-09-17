#!/usr/bin/env python
"""Evaluate 21-candidate teacher-forced ranking and hard routing cost."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from adarank_rec.config import AdaRankConfig
from adarank_rec.data import AmazonLeaveOneOut, collate_requests
from adarank_rec.metrics import hard_cost, ranking_metrics
from adarank_rec.model import AdaRankRec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--sids", required=True)
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--candidates", required=True, help=".pt [N,21], positive is column 0")
    ap.add_argument("--budget", type=float, default=.47); ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); args = ap.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = AdaRankConfig(**saved["config"]); raw = json.loads(Path(args.sids).read_text()); max_item = max(map(int, raw))
    sid = torch.zeros(max_item+1, 4, dtype=torch.long)
    for key, value in raw.items(): sid[int(key)] = torch.tensor(value)
    model = AdaRankRec(max_item, sid, cfg).to(args.device); model.load_state_dict(saved["state_dict"]); model.eval()
    candidates = torch.load(args.candidates, map_location="cpu", weights_only=True)
    loader = DataLoader(AmazonLeaveOneOut(args.data, "test"), batch_size=args.batch_size, collate_fn=collate_requests)
    all_scores, all_routes, offset = [], [], 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}; n = batch["target"].numel()
            current = candidates[offset:offset+n].to(args.device); offset += n
            all_scores.append(model.score_candidates(batch, current, args.budget).cpu())
            all_routes.append(model.controller_logits(batch, args.budget).argmax(-1).cpu())
    scores, routes = torch.cat(all_scores), torch.cat(all_routes)
    print({**ranking_metrics(scores), **hard_cost(routes, cfg.ranks, args.budget)})


if __name__ == "__main__": main()
