"""Достроить A100-прогоны до формата runs/<tag>, чтобы они попали в общий HTML.

Скачанные из ClearML артефакты содержат только историю, метрики и веса. Общая
визуализация требует ещё config.json и снапшот диагностики, посчитанный на той же
projection basis, что и Mac-прогоны, поэтому снапшот считается здесь на CPU.
"""

import json
import shutil
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import data
import diag
from train import Config as TrainConfig
import viz

SOURCE = Path("runs/a100")
PROJECTION = Path("runs/clean-projection.pt")
TAGS = ["baseline-a100", "huginn-a100", "antisymmetric-a100", "controller-a100-50M"]


def materialise(tag, block, tok):
    history = SOURCE / f"{tag}-history"
    weights = SOURCE / f"{tag}-best"
    if not (history.exists() and weights.exists()):
        print(f"[skip] {tag}: артефактов ещё нет")
        return None

    run = Path("runs") / tag
    run.mkdir(parents=True, exist_ok=True)
    shutil.copy(history, run / "history.json")
    shutil.copy(SOURCE / f"{tag}-metrics", run / "metrics.jsonl")
    shutil.copy(weights, run / "best.pt")

    model, blob = diag.load(run / "best.pt", "cpu")
    (run / "config.json").write_text(json.dumps({
        "model": blob["model_config"],
        "training": blob["training_config"],
        "runtime": blob["runtime"],
    }, indent=2) + "\n")

    torch.manual_seed(10_000 + blob["step"])
    # Быстрое CPU-ядро внимания не поддерживает forward-mode AD, который нужен диагностике.
    with sdpa_kernel(SDPBackend.MATH):
        diag.write_snapshot(model, block, tok, run, blob["step"], "cpu", PROJECTION,
                            snapshot_name="diag-best")
    rows = json.loads((run / "history.json").read_text())
    best = min(rows, key=lambda row: row["loss"])
    print(f"[ok] {tag}: step {blob['step']}, tokens {best['tokens']:,}, loss {best['loss']:.4f}")
    return tag


def main():
    tok = data.tokenizer()
    prepared, _ = data.prepare(tok, data.Config())
    block = prepared["selection"][0]
    ready = [tag for tag in TAGS if materialise(tag, block, tok)]

    # Итоговая страница показывает только финальное сравнение, общая — каждый прогон,
    # у которого есть и конфиг, и снапшот диагностики, включая smoke и cycle probe.
    viz.render(Path("runs/report.html"), ["baseline-clean-mac", "huginn-clean-mac"] + ready)
    everything = sorted(
        run.name for run in Path("runs").iterdir()
        if (run / "config.json").exists() and (run / "diag-best.json").exists()
        and "smoke" not in run.name
    )
    viz.render(Path("runs/all-experiments.html"), everything, include_legacy=True)
    print("итоговая страница:", ready)
    print("общая страница:", len(everything), "прогонов")


if __name__ == "__main__":
    main()
