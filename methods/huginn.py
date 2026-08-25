"""Huginn: concat-adapter, общий core, RMSNorm на выходе core."""

from dataclasses import asdict
import math

import torch
from torch import nn
import torch.nn.functional as F

from blocks import Block, RMSNorm, rope, truncated_bp
from config import Config

class HuginnLoopedLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        if not (cfg.n_prelude and cfg.n_core and cfg.n_coda):
            raise ValueError("Huginn needs prelude, recurrent core, and coda")

        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.prelude = nn.ModuleList(Block(cfg) for _ in range(cfg.n_prelude))
        self.core = nn.ModuleList(Block(cfg) for _ in range(cfg.n_core))
        self.coda = nn.ModuleList(Block(cfg) for _ in range(cfg.n_coda))
        self.adapter = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)
        self.core_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self._initialize()

    def _initialize(self):
        depth = self.cfg.n_prelude + self.cfg.mean_recurrence * self.cfg.n_core + self.cfg.n_coda
        std = math.sqrt(2 / (5 * self.cfg.d_model))
        out_std = math.sqrt(1 / (5 * self.cfg.d_model * depth))

        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                scale = out_std if name.endswith(("attn.proj", "down")) else std
                nn.init.trunc_normal_(module.weight, std=scale, a=-3 * scale, b=3 * scale)

    def encode(self, idx: torch.Tensor):
        cos, sin = rope(idx.shape[1], self.cfg.head_dim, self.cfg.rope_theta, idx.device)
        embedded = self.embed(idx) * math.sqrt(self.cfg.d_model)

        for block in self.prelude:
            embedded = block(embedded, cos, sin)

        return embedded, cos, sin

    def initial_state(self, embedded: torch.Tensor) -> torch.Tensor:
        state = torch.empty_like(embedded)
        std = math.sqrt(2 / 5)
        return nn.init.trunc_normal_(state, std=std, a=-3 * std, b=3 * std)

    def recurrent_step(self, state, embedded, cos, sin, trace=None, routing=None):
        state = self.adapter(torch.cat([state, embedded], -1))

        if trace is not None:
            trace.append(("concat adapter", state))

        for index, block in enumerate(self.core):
            state = block(state, cos, sin)

            if trace is not None:
                trace.append((f"core block {index}", state))

        state = self.core_norm(state)

        if trace is not None:
            trace.append(("core RMSNorm", state))

        return state

    def states(self, idx, steps, backprop_last=None, initial_state=None, trace=None):
        embedded, cos, sin = self.encode(idx)
        state = self.initial_state(embedded) if initial_state is None else initial_state
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
        sigma = 0.5
        mu = math.log(self.cfg.mean_recurrence) - sigma * sigma / 2
        rate = torch.empty(1).normal_(mu, sigma, generator=generator).exp()
        return int(torch.poisson(rate, generator=generator).item()) + 1

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
