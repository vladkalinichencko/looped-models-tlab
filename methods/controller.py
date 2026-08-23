"""Контроллер: четыре блока предлагают обновление, softmax их смешивает."""

import torch


def step(model, state, embedded, cos, sin, trace=None, routing=None):
    if routing is None:
        raise ValueError("controller step needs routing weights")
    proposals = torch.stack([block(state, cos, sin) for block in model.core], 2)
    state = (routing[..., None] * proposals).sum(2)
    if trace is not None:
        trace.append(("softmax routed blocks", state))
    return state
