"""Антисимметричный переход: Qwen-core, затем кососимметричное вращение."""

import torch
import torch.nn.functional as F


def step(model, state, embedded, cos, sin, trace=None, routing=None):
    core_out = state
    for index, block in enumerate(model.core):
        core_out = block(core_out, cos, sin)
        if trace is not None:
            trace.append((f"core block {index}", core_out))
    rotation = F.linear(core_out, model.skew - model.skew.T)
    gate = torch.sigmoid(model.skew_gate(torch.cat([state, embedded], -1)))
    state = (1 - torch.sigmoid(model.skew_decay_logit)) * state + gate * rotation
    if trace is not None:
        trace.append(("antisymmetric update", state))
    return state
