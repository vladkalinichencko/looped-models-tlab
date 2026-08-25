"""Контроллер: четыре блока предлагают обновление, softmax их смешивает."""

from dataclasses import asdict

import torch
from torch import nn
import torch.nn.functional as F

from blocks import Block, RMSNorm, init_normal, rope, truncated_bp
from config import Config

class ControllerLoopedLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        if cfg.n_core != 4:
            raise ValueError("controller experiment uses four routed blocks")

        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.prelude = nn.ModuleList(Block(cfg) for _ in range(cfg.n_prelude))
        self.core = nn.ModuleList(Block(cfg) for _ in range(cfg.n_core))
        self.coda = nn.ModuleList(Block(cfg) for _ in range(cfg.n_coda))
        self.controller_init = nn.Linear(cfg.d_model, cfg.d_model)
        self.controller_cell = nn.GRUCell(4, cfg.d_model)
        self.controller_head = nn.Linear(cfg.d_model, 4)
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

    def controller_routing(self, embedded: torch.Tensor, steps: int):
        hidden = torch.tanh(self.controller_init(embedded.detach())).flatten(0, 1)
        action = embedded.new_zeros(hidden.shape[0], 4)
        rows = []

        for _ in range(steps):
            hidden = self.controller_cell(action, hidden)
            logits = self.controller_head(hidden).view(*embedded.shape[:2], 4)
            action = logits.softmax(-1).flatten(0, 1)
            rows.append((logits, action.view(*embedded.shape[:2], 4)))

        return rows

    def recurrent_step(self, state, embedded, cos, sin, trace=None, routing=None):
        proposals = torch.stack([block(state, cos, sin) for block in self.core], 2)
        state = (routing[..., None] * proposals).sum(2)

        if trace is not None:
            trace.append(("softmax routed blocks", state))

        return state

    def states(self, idx, steps, backprop_last=None, initial_state=None, trace=None):
        embedded, cos, sin = self.encode(idx)
        state = embedded if initial_state is None else initial_state
        routing = self.controller_routing(embedded, steps)
        limit = self.cfg.backprop_last if backprop_last is None else backprop_last
        keep = steps if limit == 0 else min(steps, limit)

        def body(state, step, layer_trace):
            return self.recurrent_step(state, embedded, cos, sin, layer_trace, routing[step][1])

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

        if targets is not None:
            return self.controller_forward(idx, targets, steps)

        state = list(self.states(idx, steps))[-1]
        logits = self.decode(state)
        loss = None
        return logits, loss

    def controller_forward(self, idx: torch.Tensor, targets: torch.Tensor, steps: int):
        embedded, cos, sin = self.encode(idx)
        state = embedded
        hidden = torch.tanh(self.controller_init(embedded.detach())).flatten(0, 1)
        action = embedded.new_zeros(hidden.shape[0], 4)
        block_losses, controller_losses = [], []

        for _ in range(steps):
            hidden = self.controller_cell(action, hidden)
            routing_logits = self.controller_head(hidden).view(*embedded.shape[:2], 4)
            proposals = torch.stack([block(state, cos, sin) for block in self.core], 2)
            branch_logits = torch.stack([self.decode(proposals[:, :, k]) for k in range(4)], 2)
            branch_ce = F.cross_entropy(
                branch_logits.reshape(-1, branch_logits.shape[-1]),
                targets[..., None].expand(-1, -1, 4).reshape(-1), reduction="none",
            ).view(*targets.shape, 4)
            oracle = branch_ce.detach().argmin(-1)
            controller_losses.append(F.cross_entropy(routing_logits.flatten(0, 1), oracle.flatten()))
            normalized = F.normalize(proposals, dim=-1)
            similarity = normalized @ normalized.transpose(-1, -2)
            repel = similarity.triu(diagonal=1).sum((-1, -2)) / 6
            block_losses.append(branch_ce.mean() + self.cfg.contrastive_weight * repel.mean())
            action = F.one_hot(oracle, 4).to(proposals.dtype)
            state = (action[..., None] * proposals).sum(2)
            action = action.flatten(0, 1)

        logits = self.decode(state)
        block_loss = torch.stack(block_losses).mean()
        controller_loss = torch.stack(controller_losses).mean()
        loss = block_loss + controller_loss
        self.last_objective = {"block_contrastive": float(block_loss.detach()),
                               "controller_teacher_forcing": float(controller_loss.detach())}
        return logits, loss

    def n_params(self):
        total = sum(p.numel() for p in self.parameters())
        return total, total - self.embed.weight.numel()

    def config_dict(self):
        return asdict(self.cfg)
