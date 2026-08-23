"""Baseline: повторяется общий core без инъекции входа и нормализации."""

from methods.huginn import run_core


def step(model, state, embedded, cos, sin, trace=None, routing=None):
    return run_core(model, state, cos, sin, trace)
