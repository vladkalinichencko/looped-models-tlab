"""Evaluate one checkpoint at fixed recurrent depths on a named clean split."""

import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

import data
import diag
from train import Config, fixed_rng, pick_device


@torch.no_grad()
def evaluate_checkpoint(checkpoint: Path, prepared, split: str = "selection",
                        recurrences=(0, 1, 2, 4, 8, 16)):
    model, blob = diag.load(checkpoint, pick_device())
    cfg = Config(**blob["training_config"])
    out = checkpoint.parent / f"eval_{split}.json"
    rows = []
    for steps in recurrences:
        with fixed_rng(str(next(model.parameters()).device), cfg.seed + 10_000):
            losses = []
            for x, y in data.batches(prepared[split], cfg.batch_size,
                                     next(model.parameters()).device):
                logits = model(x, steps=steps)[0]
                losses.append(F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).item())
        loss = sum(losses) / len(losses)
        rows.append({"recurrence": steps, "loss": loss, "ppl": math.exp(loss)})
        out.write_text(json.dumps({"checkpoint": str(checkpoint), "split": split,
                                   "rows": rows}, indent=2) + "\n")
        print(rows[-1], flush=True)
    return rows


if __name__ == "__main__":
    checkpoint = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/huginn-preliminary/best.pt")
    _, blob = diag.load(checkpoint)
    train_cfg = Config(**blob["training_config"])
    prepared, _ = data.prepare(data.tokenizer(), data.Config(
        seq_len=train_cfg.seq_len, batch_size=train_cfg.batch_size, train_tokens=train_cfg.tokens))
    evaluate_checkpoint(checkpoint, prepared)
