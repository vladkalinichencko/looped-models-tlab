"""Val perplexity of a checkpoint, optionally swept over the number of loops.

    python eval.py runs/loop4/ckpt.pt
    python eval.py runs/loop4/ckpt.pt --loops 1 2 4 8 16 32

The loop sweep is the interesting plot for this task: where does the extra compute
stop paying off, and does it ever start hurting?
"""

import argparse
import json
import math
import os
import pathlib

import torch

import diag
from train import evaluate, pick_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--loops", type=int, nargs="+", default=None)
    p.add_argument("--val-batches", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = pick_device(args.device)
    model, blob = diag.load(args.ckpt, device)
    cfg = model.cfg
    # те же батчи, что и у диагностики, из кэша: иначе каждый эвал заново тянет FineWeb
    val = diag.val_batches(blob, cfg, args.batch_size, args.val_batches, device)

    rows = []
    for n in args.loops or [cfg.n_loops]:
        loss = evaluate(model, val, n_loops=n)
        rows.append({"n_loops": n, "val_loss": loss, "val_ppl": math.exp(loss)})
        print(f"loops={n:>3}  val loss {loss:.4f}  ppl {math.exp(loss):.2f}")

    out = pathlib.Path(args.ckpt).parent / "eval.json"
    out.write_text(json.dumps({"ckpt": args.ckpt, "trained_loops": cfg.n_loops,
                               "rows": rows}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
    os._exit(0)
