"""Minimal Qwen3-style decoder whose block stack is applied `n_loops` times.

Qwen3-isms kept on purpose: RMSNorm pre-norm, SwiGLU, RoPE, per-head QK-norm.
Everything else is stripped so the looping scheme stays the only moving part.
"""

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 4  # layers inside ONE loop block
    n_loops: int = 4
    d_ff: int = 896
    max_seq: int = 512
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    loop_scheme: str = "stack"  # см. loop_plan
    group_size: int = 2  # размер группы для loop_scheme="group"
    loop_norm: bool = False  # нормализовать h после каждого лупа (см. NOTES: рост нормы)
    input_injection: bool = False  # подмешивать эмбеддинги на каждом лупе (Huginn)
    step_cond: bool = False  # прибавлять эмбеддинг номера шага (Universal Transformers)
    grad_checkpoint: bool = False  # хранить только состояния на границах шагов


def loop_plan(n_layers, n_loops, scheme="stack", group_size=2):
    """Порядок применения блоков: список шагов, шаг — список индексов блоков.

    stack — f1 f2 f1 f2 (так делают Huginn и Ouro), layer — f1 f1 f2 f2, group —
    (f1 f2)(f1 f2)(f3 f4)(f3 f4). Число применений блока одинаково у всех трёх,
    различается только порядок, так что сравнение схем идёт при равном компьюте.
    """
    size = {"stack": n_layers, "layer": 1}.get(scheme, group_size)
    return [list(range(s, min(s + size, n_layers)))
            for s in range(0, n_layers, size) for _ in range(n_loops)]


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * x).to(dtype)


def rope_cache(seq_len, head_dim, theta, device):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len, device=device).float(), inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):  # x: (b, h, s, hd)
    cos, sin = cos[None, None].to(x.dtype), sin[None, None].to(x.dtype)
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, cos, sin):
        b, s, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(b, s, self.n_heads, self.head_dim).transpose(1, 2) for t in (q, k, v))
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(o.transpose(1, 2).reshape(b, s, -1))


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))


class LoopedLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.loop_norm = RMSNorm(cfg.d_model) if cfg.loop_norm else None
        self.step_emb = nn.Embedding(len(self.plan(cfg.n_loops)), cfg.d_model) if cfg.step_cond else None
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.apply(self._init)

        # GPT-2 residual init, но глубина здесь — число применений блока за проход,
        # а не число блоков: при 16 лупах в поток невязки пишут 16 раз одни и те же
        # веса. Без этого ‖h‖ растёт тем сильнее, чем больше лупов, и сравнение
        # схем превращается в сравнение масштабов активаций.
        depth = 2 * sum(len(step) for step in self.plan())
        for block in self.blocks:
            for w in (block.attn.proj.weight, block.mlp.down.weight):
                w.data.mul_(depth ** -0.5)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def plan(self, n_loops=None):
        return loop_plan(self.cfg.n_layers, n_loops or self.cfg.n_loops,
                         self.cfg.loop_scheme, self.cfg.group_size)

    def trace(self, idx, n_loops=None):
        """h после эмбеддингов и после каждого шага плана.

        Ровно то же вычисление, что и в forward — diag.py считает метрики по этому
        генератору, так что диагностика не может разойтись с обучением.
        """
        cos, sin = rope_cache(idx.shape[1], self.cfg.d_model // self.cfg.n_heads,
                              self.cfg.rope_theta, idx.device)
        h = h0 = self.embed(idx)
        yield h
        for t, step in enumerate(self.plan(n_loops)):
            if self.cfg.input_injection:
                h = h + h0
            if self.step_emb is not None:
                h = h + self.step_emb.weight[t]

            def apply(x, step=step):
                for i in step:
                    x = self.blocks[i](x, cos, sin)
                return x

            # активации внутри шага пересчитываются при бэкпропе вместо хранения:
            # при 16 лупах граф держит 64 применения блока и упирается в память
            h = (torch.utils.checkpoint.checkpoint(apply, h, use_reentrant=False)
                 if self.cfg.grad_checkpoint and self.training and h.requires_grad else apply(h))
            if self.loop_norm is not None:
                h = self.loop_norm(h)
            yield h

    def head(self, h):
        return self.lm_head(self.norm(h))

    def forward(self, idx, targets=None, n_loops=None):
        for h in self.trace(idx, n_loops):
            pass
        logits = self.head(h)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def n_params(self):
        """(total, non-embedding) — the budget reading matters, see NOTES."""
        total = sum(p.numel() for p in self.parameters())
        emb = self.embed.weight.numel()
        if not self.cfg.tie_embeddings:
            emb += self.lm_head.weight.numel()
        return total, total - emb

    def config_dict(self):
        return asdict(self.cfg)
