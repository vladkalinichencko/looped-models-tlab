import torch
from torch import nn
import torch.nn.functional as F

from config import Config

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
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=self.n_heads != self.n_kv_heads)
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

def init_normal(module: nn.Module):
    for child in module.modules():
        if isinstance(child, (nn.Linear, nn.Embedding)):
            nn.init.normal_(child.weight, std=0.02)

            if isinstance(child, nn.Linear) and child.bias is not None:
                nn.init.zeros_(child.bias)

def truncated_bp(state, steps, keep, body, trace=None):
    if keep == steps and not state.requires_grad:
        state = state.detach().requires_grad_(True)

    yield state

    for step in range(steps):
        if step < steps - keep:
            with torch.no_grad():
                layer_trace = [] if trace is not None else None
                state = body(state, step, layer_trace)

            if step + 1 == steps - keep:
                state = state.detach().requires_grad_(True)

        else:
            layer_trace = [] if trace is not None else None
            state = body(state, step, layer_trace)

        if trace is not None:
            trace.extend((step + 1, name, value) for name, value in layer_trace)

        yield state
