"""Qwen3 blocks and the two models used in the first clean comparison."""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    vocab_size: int
    method: str = "baseline"  # baseline | huginn
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

    def __post_init__(self):
        if self.method not in {"baseline", "huginn"}:
            raise ValueError(f"unknown method: {self.method}")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.method == "huginn" and not (self.n_prelude and self.n_core and self.n_coda):
            raise ValueError("Huginn needs prelude, recurrent core, and coda")


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(dtype)


def rope(seq_len: int, head_dim: int, theta: float, device: torch.device):
    inv = theta ** (-torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    phase = torch.outer(torch.arange(seq_len, device=device).float(), inv)
    phase = torch.cat([phase, phase], -1)
    return phase.cos()[None, None], phase.sin()[None, None]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    left, right = x.chunk(2, -1)
    return torch.cat([-right, left], -1)


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.q = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = self.q_norm(self.q(x).view(batch, seq, self.n_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k(x).view(batch, seq, self.n_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v(x).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                            enable_gqa=self.n_heads != self.n_kv_heads)
        return self.proj(y.transpose(1, 2).reshape(batch, seq, -1))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        y = self.mlp_norm(x)
        return x + self.down(F.silu(self.gate(y)) * self.up(y))


class LoopedLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.prelude = nn.ModuleList(Block(cfg) for _ in range(cfg.n_prelude))
        self.core = nn.ModuleList(Block(cfg) for _ in range(cfg.n_core))
        self.coda = nn.ModuleList(Block(cfg) for _ in range(cfg.n_coda))
        self.adapter = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False) if cfg.method == "huginn" else None
        self.core_norm = RMSNorm(cfg.d_model, cfg.norm_eps) if cfg.method == "huginn" else None
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self._initialize()

    def _initialize(self):
        if self.cfg.method == "baseline":
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    nn.init.normal_(module.weight, std=0.02)
            return
        depth = self.cfg.n_prelude + self.cfg.mean_recurrence * self.cfg.n_core + self.cfg.n_coda
        std = math.sqrt(2 / (5 * self.cfg.d_model))
        out_std = math.sqrt(1 / (5 * self.cfg.d_model * depth))
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                scale = out_std if name.endswith(("attn.proj", "down")) else std
                nn.init.trunc_normal_(module.weight, std=scale, a=-3 * scale, b=3 * scale)

    @property
    def embedding_scale(self) -> float:
        return math.sqrt(self.cfg.d_model) if self.cfg.method == "huginn" else 1.0

    def encode(self, idx: torch.Tensor):
        cos, sin = rope(idx.shape[1], self.cfg.head_dim, self.cfg.rope_theta, idx.device)
        embedded = self.embed(idx) * self.embedding_scale
        for block in self.prelude:
            embedded = block(embedded, cos, sin)
        return embedded, cos, sin

    def initial_state(self, embedded: torch.Tensor) -> torch.Tensor:
        if self.cfg.method == "baseline":
            return embedded
        state = torch.empty_like(embedded)
        std = math.sqrt(2 / 5)
        return nn.init.trunc_normal_(state, std=std, a=-3 * std, b=3 * std)

    def recurrent_step(self, state: torch.Tensor, embedded: torch.Tensor,
                       cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        if self.adapter is not None:
            state = self.adapter(torch.cat([state, embedded], -1))
        for block in self.core:
            state = block(state, cos, sin)
        return self.core_norm(state) if self.core_norm is not None else state

    def states(self, idx: torch.Tensor, steps: int, backprop_last: int | None = None,
               initial_state: torch.Tensor | None = None):
        embedded, cos, sin = self.encode(idx)
        state = self.initial_state(embedded) if initial_state is None else initial_state
        limit = self.cfg.backprop_last if backprop_last is None else backprop_last
        keep = steps if limit == 0 else min(steps, limit)
        if keep == steps and not state.requires_grad:
            state = state.detach().requires_grad_(True)
        yield state
        for step in range(steps):
            if step < steps - keep:
                with torch.no_grad():
                    state = self.recurrent_step(state, embedded, cos, sin)
                if step + 1 == steps - keep:
                    state = state.detach().requires_grad_(True)
            else:
                state = self.recurrent_step(state, embedded, cos, sin)
            yield state

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        cos, sin = rope(state.shape[1], self.cfg.head_dim, self.cfg.rope_theta, state.device)
        for block in self.coda:
            state = block(state, cos, sin)
        return self.lm_head(self.final_norm(state))

    def sample_steps(self, generator: torch.Generator) -> int:
        if self.cfg.method == "baseline":
            return 1
        sigma = 0.5
        mu = math.log(self.cfg.mean_recurrence) - sigma * sigma / 2
        rate = torch.empty(1).normal_(mu, sigma, generator=generator).exp()
        return int(torch.poisson(rate, generator=generator).item()) + 1

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                steps: int | None = None):
        steps = steps or self.cfg.mean_recurrence
        state = list(self.states(idx, steps))[-1]
        logits = self.decode(state)
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss

    def n_params(self):
        total = sum(p.numel() for p in self.parameters())
        return total, total - self.embed.weight.numel()

    def config_dict(self):
        return asdict(self.cfg)
