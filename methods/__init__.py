"""Один файл на схему лупинга; LoopedLM.recurrent_step только выбирает нужную."""

from methods import antisymmetric, controller, huginn, plain

STEPS = {
    "baseline": plain.step,
    "huginn": huginn.step,
    "antisymmetric": antisymmetric.step,
    "controller": controller.step,
}
