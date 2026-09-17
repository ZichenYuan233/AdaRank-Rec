#!/usr/bin/env python
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from adarank_rec.config import AdaRankConfig
from adarank_rec.data import AmazonLeaveOneOut, collate_requests
from adarank_rec.model import AdaRankRec
from adarank_rec.trainer import AdaRankTrainer


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--sids", required=True)
    ap.add_argument("--out", default="checkpoints/adarank.pt"); ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); args = ap.parse_args()
    cfg = AdaRankConfig(); random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    raw = json.loads(Path(args.sids).read_text()); max_item = max(map(int, raw)); sid = torch.zeros(max_item + 1, 4, dtype=torch.long)
    for key, value in raw.items(): sid[int(key)] = torch.tensor(value)
    train = DataLoader(AmazonLeaveOneOut(args.data, "train"), batch_size=args.batch_size, shuffle=True, collate_fn=collate_requests)
    model = AdaRankRec(max_item, sid, cfg).to(args.device); trainer = AdaRankTrainer(model, torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay))
    for stage, epochs in ((1, cfg.stage1_epochs), (2, cfg.stage2_epochs), (3, cfg.stage3_epochs)):
        model.train()
        for epoch in range(epochs):
            for batch in train:
                stat = trainer.stage1_step(batch) if stage == 1 else trainer.stage2_step(batch) if stage == 2 else trainer.stage3_step(batch, epoch, epochs)
            print(f"stage={stage} epoch={epoch+1}/{epochs} {stat}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg.__dict__, "state_dict": model.state_dict()}, args.out)


if __name__ == "__main__": main()
