#!/usr/bin/env python
"""Create chronological Amazon-2014 leave-one-out JSONL requests and SIDs.

Input interactions are JSON lines with reviewerID, asin and unixReviewTime.
Metadata is JSON lines keyed by asin and includes precomputed `visual` and
`text` vectors.  The cached-feature protocol follows the paper.
"""
from __future__ import annotations
import argparse, gzip, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from adarank_rec.sid import build_sids


def lines(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", required=True); ap.add_argument("--metadata", required=True)
    ap.add_argument("--output", required=True); ap.add_argument("--visual-dim", type=int, default=768)
    ap.add_argument("--text-dim", type=int, default=768); ap.add_argument("--max-history", type=int, default=50)
    args = ap.parse_args()
    reviews = list(lines(args.reviews))
    # Official 5-core iterative filtering.
    changed = True
    while changed:
        users = Counter(r["reviewerID"] for r in reviews); items = Counter(r["asin"] for r in reviews)
        kept = [r for r in reviews if users[r["reviewerID"]] >= 5 and items[r["asin"]] >= 5]
        changed = len(kept) != len(reviews); reviews = kept
    meta = {x["asin"]: x for x in lines(args.metadata) if "asin" in x}
    item_ids = sorted({r["asin"] for r in reviews}); item_to_index = {x: i + 1 for i, x in enumerate(item_ids)}
    def vector(item, key, dim):
        x = meta.get(item, {}).get(key)
        return np.zeros(dim, np.float32) if x is None else np.asarray(x, np.float32)
    # Fused cached representation used solely to construct residual SIDs.
    sid_vectors = np.stack([np.concatenate([vector(x, "visual", args.visual_dim), vector(x, "text", args.text_dim)]) for x in item_ids])
    sids = build_sids(sid_vectors, item_ids)
    by_user = defaultdict(list)
    for r in reviews: by_user[r["reviewerID"]].append(r)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for events in by_user.values():
            events.sort(key=lambda r: (r["unixReviewTime"], r["asin"]))
            ids = [item_to_index[r["asin"]] for r in events]
            for i in range(1, len(events)):
                split = "test" if i == len(events)-1 else "valid" if i == len(events)-2 else "train"
                target_item = events[i]["asin"]
                row = {"split": split, "history": ids[max(0, i-args.max_history):i], "target": ids[i],
                       "visual": vector(target_item, "visual", args.visual_dim).tolist(),
                       "text": vector(target_item, "text", args.text_dim).tolist()}
                f.write(json.dumps(row) + "\n")
    (output.parent / "item_to_index.json").write_text(json.dumps(item_to_index), encoding="utf-8")
    sid_by_index = {str(item_to_index[x]): sid for x, sid in sids.items()}
    (output.parent / "sid_table.json").write_text(json.dumps(sid_by_index), encoding="utf-8")

if __name__ == "__main__": main()
