"""Chronological leave-one-out data and cached modality feature loading."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence
import torch
from torch.utils.data import Dataset


@dataclass
class Request:
    history: List[int]
    target: int
    visual: torch.Tensor
    text: torch.Tensor


class AmazonLeaveOneOut(Dataset):
    """JSONL requests created by scripts/prepare_amazon.py.

    Rows contain `history`, `target`, `visual`, and `text`; feature arrays can
    also be stored in separate .pt files and addressed by item index.
    """
    def __init__(self, path: str | Path, split: str):
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
        self.rows = [r for r in rows if r["split"] == split]

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {
            "history": torch.tensor(row["history"], dtype=torch.long),
            "target": torch.tensor(row["target"], dtype=torch.long),
            "visual": torch.tensor(row["visual"], dtype=torch.float),
            "text": torch.tensor(row["text"], dtype=torch.float),
        }


def collate_requests(rows: Sequence[dict]) -> dict:
    max_len = max(x["history"].numel() for x in rows)
    history = torch.zeros(len(rows), max_len, dtype=torch.long)
    mask = torch.zeros(len(rows), max_len, dtype=torch.bool)
    for i, row in enumerate(rows):
        n = row["history"].numel()
        history[i, :n] = row["history"]
        mask[i, :n] = True
    return {"history": history, "history_mask": mask,
            "target": torch.stack([x["target"] for x in rows]),
            "visual": torch.stack([x["visual"] for x in rows]),
            "text": torch.stack([x["text"] for x in rows])}
