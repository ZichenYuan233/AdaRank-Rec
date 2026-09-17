#!/usr/bin/env python
"""Generate the paper's deterministic 20-negative candidate pools."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--num-items", type=int, required=True)
    ap.add_argument("--output", default="test_candidates.pt"); ap.add_argument("--seed", type=int, default=42); args = ap.parse_args()
    generator = torch.Generator().manual_seed(args.seed); pools = []
    for line in Path(args.data).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["split"] != "test": continue
        forbidden = set(row["history"]) | {row["target"]}
        available = torch.tensor([i for i in range(1, args.num_items + 1) if i not in forbidden])
        if len(available) < 20: raise ValueError("fewer than 20 eligible negatives")
        negative = available[torch.randperm(len(available), generator=generator)[:20]]
        pools.append(torch.cat([torch.tensor([row["target"]]), negative]))
    torch.save(torch.stack(pools), args.output)


if __name__ == "__main__": main()
