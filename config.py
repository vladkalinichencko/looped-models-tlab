from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    vocab_size: int
    method: str = "baseline"
    d_model: int = 384
    n_heads: int = 6
    n_kv_heads: int = 3
    head_dim: int = 128
    d_ff: int = 1152
    n_prelude: int = 0
    n_core: int = 4
    n_coda: int = 0
    mean_recurrence: int = 1
    backprop_last: int = 0
    max_seq: int = 512
    rope_theta: float = 1_000_000.0
    norm_eps: float = 1e-6
    contrastive_weight: float = 0.01
    antisymmetric_decay: float = 0.0

    def __post_init__(self):
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")

        if not 0 <= self.antisymmetric_decay < 1:
            raise ValueError("antisymmetric_decay must be in [0, 1)")
