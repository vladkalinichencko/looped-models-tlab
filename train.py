"""Pretrain the looped LM on FineWeb under a fixed token budget.

    python train.py --tag loop4 --n-loops 4 --tokens 100_000_000
    python train.py --tag smoke --tokens 200_000 --eval-every 50   # sanity run

Every variant must share tokens/seed/tokenizer with its baseline — the looping
scheme is supposed to be the only difference.
"""

import argparse
import json
import math
import pathlib
import time

import torch

import data
from model import Config, LoopedLM


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def lr_at(step, total, warmup, lr, min_lr):
    if step < warmup:
        return lr * (step + 1) / warmup
    t = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * min(t, 1.0)))


@torch.no_grad()
def evaluate(model, val, n_loops=None):
    model.eval()
    loss = sum(model(x, y, n_loops=n_loops)[1].item() for x, y in val) / len(val)
    model.train()
    return loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    p.add_argument("--tokens", type=int, default=100_000_000, help="training token budget")
    p.add_argument("--n-loops", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2, help="layers inside one loop block")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=704)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--min-lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-batches", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    out = pathlib.Path("runs") / args.tag
    out.mkdir(parents=True, exist_ok=True)

    tok = data.tokenizer(args.tokenizer)
    cfg = Config(
        vocab_size=len(tok),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_loops=args.n_loops,
        d_ff=args.d_ff,
        max_seq=args.seq_len,
    )
    model = LoopedLM(cfg).to(device)
    total, non_emb = model.n_params()
    print(f"device={device}  params: {total/1e6:.2f}M total / {non_emb/1e6:.2f}M non-embedding")
    if total > 10e6:
        print("!! over the 10M budget on the total-parameter reading — see NOTES")

    val, train = data.split(tok, args.seq_len, args.batch_size, args.val_batches, device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)
    tokens_per_step = args.batch_size * args.seq_len
    total_steps = args.tokens // tokens_per_step
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else None

    best, history, t0 = float("inf"), [], time.time()
    for step, (x, y) in enumerate(train):
        if step >= total_steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total_steps, args.warmup, args.lr, args.min_lr)

        if amp is not None:
            with amp:
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % args.log_every == 0:
            seen = step * tokens_per_step
            print(f"step {step:>6}/{total_steps}  tok {seen/1e6:6.1f}M  "
                  f"loss {loss.item():.4f}  {time.time()-t0:.0f}s")

        if (step and step % args.eval_every == 0) or step == total_steps - 1:
            vl = evaluate(model, val)
            history.append({"step": step, "tokens": step * tokens_per_step,
                            "val_loss": vl, "val_ppl": math.exp(vl)})
            print(f"  val loss {vl:.4f}  ppl {math.exp(vl):.2f}")
            if vl < best:
                best = vl
                torch.save({"cfg": cfg.__dict__, "model": model.state_dict(),
                            "step": step, "val_loss": vl, "tokenizer": args.tokenizer},
                           out / "ckpt.pt")

    best_ppl = math.exp(best) if history else None
    (out / "history.json").write_text(json.dumps(
        {"config": vars(args), "params": {"total": total, "non_embedding": non_emb},
         "best_val_loss": best if history else None, "best_val_ppl": best_ppl,
         "history": history}, indent=2))
    print(f"best val ppl {best_ppl} -> {out}")


if __name__ == "__main__":
    main()
