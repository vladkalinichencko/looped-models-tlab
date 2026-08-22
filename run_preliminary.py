"""Run the clean Mac comparison sequentially with fixed Python configurations."""

import json
from pathlib import Path

import torch

import data
import diag
from model import Config as ModelConfig, LoopedLM
from train import Config as TrainConfig, train
import viz


DATA = data.Config()
RUNS = [
    (
        ModelConfig(vocab_size=16_384, n_core=4),
        TrainConfig(tag="baseline-clean-mac"),
    ),
    (
        ModelConfig(vocab_size=16_384, method="huginn", n_prelude=1, n_core=2, n_coda=1,
                    mean_recurrence=4, backprop_last=4),
        TrainConfig(tag="huginn-clean-mac"),
    ),
]


def completed(tag: str) -> bool:
    out = Path("runs") / tag
    if not all((out / name).exists() for name in ("run.json", "best.pt", "last.pt", "diag.jsonl")):
        return False
    return json.loads((out / "run.json").read_text()).get("status") == "completed"


def run_all():
    tok = data.tokenizer()
    prepared, manifest = data.prepare(tok, DATA)
    for model_cfg, train_cfg in RUNS:
        model_cfg = ModelConfig(**{**model_cfg.__dict__, "vocab_size": len(tok)})
        if not completed(train_cfg.tag):
            torch.manual_seed(train_cfg.seed)
            result = train(LoopedLM(model_cfg), prepared, manifest, tok, train_cfg)
            print(result)
        if train_cfg.tag == "baseline-clean-mac" and not Path("runs/projection.pt").exists():
            history = json.loads((Path("runs") / train_cfg.tag / "history.json").read_text())
            best_step = min(history, key=lambda row: row["loss"])["step"]
            diag.fit_projection(Path("runs") / train_cfg.tag / "snapshots" /
                                f"step{best_step:06d}.pt")
        viz.render(Path("runs") / train_cfg.tag / "report.html", [train_cfg.tag])
        viz.render()


if __name__ == "__main__":
    run_all()
