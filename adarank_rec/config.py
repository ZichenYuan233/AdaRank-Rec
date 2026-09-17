from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class AdaRankConfig:
    # Paper defaults
    dim: int = 256
    decoder_dim: int = 3584
    num_queries: int = 8
    max_history: int = 50
    ranks: Tuple[int, ...] = (32, 64, 128, 256)
    num_sid_levels: int = 3
    codes_per_level: int = 256
    visual_dim: int = 768
    text_dim: int = 768
    dropout: float = 0.1
    controller_hidden: int = 512
    # Optimization defaults
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    orth_weight: float = 1e-3
    consistency_weight: float = 0.1
    dual_lr: float = 0.01
    epsilon: float = 0.01
    budgets: Tuple[float, ...] = (0.35, 0.47, 0.60)
    stage1_epochs: int = 10
    stage2_epochs: int = 3
    stage3_epochs: int = 5
    seed: int = 42

    @property
    def max_rank(self) -> int:
        return self.ranks[-1]
