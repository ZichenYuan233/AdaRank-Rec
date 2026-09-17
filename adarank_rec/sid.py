"""Residual-quantized semantic identifiers with deterministic collisions."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple
import numpy as np


def build_sids(item_vectors: np.ndarray, item_ids: Sequence[str], n_codes: int = 256,
               levels: int = 3, seed: int = 42) -> Dict[str, Tuple[int, int, int, int]]:
    """Fit residual k-means and return fixed-length, 4-token SIDs.

    The fourth token is the deterministic (item-id sorted) position among items
    whose first three codes collide.  Each level occupies a disjoint vocabulary
    range, so the sequence is unambiguous to an autoregressive decoder.
    """
    if len(item_ids) != len(item_vectors):
        raise ValueError("item_ids and item_vectors must have equal length")
    if levels != 3:
        raise ValueError("The paper SID has exactly three residual code levels")
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError as exc:
        raise ImportError("Install scikit-learn to construct SIDs") from exc

    residual = np.asarray(item_vectors, dtype=np.float32).copy()
    assignments: List[np.ndarray] = []
    for level in range(levels):
        km = MiniBatchKMeans(n_clusters=n_codes, random_state=seed + level,
                             batch_size=min(4096, len(residual)), n_init="auto")
        labels = km.fit_predict(residual)
        assignments.append(labels)
        residual -= km.cluster_centers_[labels]

    triples = list(zip(*assignments))
    buckets: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for index, triple in enumerate(triples):
        buckets[tuple(map(int, triple))].append(index)
    collision = np.zeros(len(item_ids), dtype=np.int64)
    for members in buckets.values():
        for position, index in enumerate(sorted(members, key=lambda i: str(item_ids[i]))):
            collision[index] = position
    max_collision = int(collision.max()) + 1
    if max_collision > n_codes:
        raise ValueError(f"collision vocabulary needs {max_collision} tokens; increase n_codes")
    return {
        str(item_id): (int(assignments[0][i]), int(assignments[1][i]) + n_codes,
                       int(assignments[2][i]) + 2 * n_codes, int(collision[i]) + 3 * n_codes)
        for i, item_id in enumerate(item_ids)
    }


def sid_vocab_size(n_codes: int = 256) -> int:
    return 4 * n_codes
