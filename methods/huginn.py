"""Huginn: concat-adapter, общий core, RMSNorm на выходе core."""

import torch


def step(model, state, embedded, cos, sin, trace=None, routing=None):
    state = model.adapter(torch.cat([state, embedded], -1))
    if trace is not None:
        trace.append(("input adapter", state))
    state = run_core(model, state, cos, sin, trace)
    state = model.core_norm(state)
    if trace is not None:
        trace.append(("core RMSNorm", state))
    return state


def run_core(model, state, cos, sin, trace=None):
    for index, block in enumerate(model.core):
        state = block(state, cos, sin)
        if trace is not None:
            trace.append((f"core block {index}", state))
    return state
