"""Evaluate one checkpoint at fixed recurrent depths on a named clean split."""

import json
import math
from pathlib import Path
import sys

import torch

import data
import diag
from train import Config, fixed_rng, pick_device


@torch.no_grad()
def evaluate_checkpoint(checkpoint: Path, prepared, split: str = "selection",
                        recurrences=(1, 2, 4, 8, 16)):
    model, blob = diag.load(checkpoint, pick_device())
    cfg = Config(**blob["training_config"])
    rows = []
    for steps in recurrences:
        with fixed_rng(str(next(model.parameters()).device), cfg.seed + 10_000):
            losses = [model(x, y, steps=steps)[1].item()
                      for x, y in data.batches(prepared[split], cfg.batch_size,
                                               next(model.parameters()).device)]
        loss = sum(losses) / len(losses)
        rows.append({"recurrence": steps, "loss": loss, "ppl": math.exp(loss)})
    out = checkpoint.parent / f"eval_{split}.json"
    out.write_text(json.dumps({"checkpoint": str(checkpoint), "split": split,
                               "rows": rows}, indent=2) + "\n")
    return rows


if __name__ == "__main__":
    checkpoint = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/huginn-preliminary/best.pt")
    _, blob = diag.load(checkpoint)
    train_cfg = Config(**blob["training_config"])
    prepared, _ = data.prepare(data.tokenizer(), data.Config(
        seq_len=train_cfg.seq_len, batch_size=train_cfg.batch_size, train_tokens=train_cfg.tokens))
    for row in evaluate_checkpoint(checkpoint, prepared):
        print(row)
