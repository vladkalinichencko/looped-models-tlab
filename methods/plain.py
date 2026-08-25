"""Baseline: повторяется общий core без инъекции входа и нормализации."""

from dataclasses import asdict

import torch
from torch import nn
import torch.nn.functional as F

from blocks import Block, RMSNorm, init_normal, rope, truncated_bp
from config import Config

class PlainLoopedLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.prelude = nn.ModuleList(Block(cfg) for _ in range(cfg.n_prelude))
        self.core = nn.ModuleList(Block(cfg) for _ in range(cfg.n_core))
        self.coda = nn.ModuleList(Block(cfg) for _ in range(cfg.n_coda))
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        init_normal(self)

    def encode(self, idx: torch.Tensor):
        cos, sin = rope(idx.shape[1], self.cfg.head_dim, self.cfg.rope_theta, idx.device)
        embedded = self.embed(idx)

        for block in self.prelude:
            embedded = block(embedded, cos, sin)

        return embedded, cos, sin

    def recurrent_step(self, state, embedded, cos, sin, trace=None, routing=None):
        for index, block in enumerate(self.core):
            state = block(state, cos, sin)

            if trace is not None:
                trace.append((f"core block {index}", state))

        return state

    def states(self, idx, steps, backprop_last=None, initial_state=None, trace=None):
        embedded, cos, sin = self.encode(idx)
        state = embedded if initial_state is None else initial_state
        limit = self.cfg.backprop_last if backprop_last is None else backprop_last
        keep = steps if limit == 0 else min(steps, limit)

        def body(state, step, layer_trace):
            return self.recurrent_step(state, embedded, cos, sin, layer_trace)

        yield from truncated_bp(state, steps, keep, body, trace)

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        cos, sin = rope(state.shape[1], self.cfg.head_dim, self.cfg.rope_theta, state.device)

        for block in self.coda:
            state = block(state, cos, sin)

        return self.lm_head(self.final_norm(state))

    def sample_steps(self, generator: torch.Generator) -> int:
        return self.cfg.mean_recurrence

    def forward(self, idx, targets=None, steps=None):
        steps = self.cfg.mean_recurrence if steps is None else steps
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
