"""Pretrain the looped LM on FineWeb under a fixed token budget.

    python train.py --tag loop4 --n-loops 4 --tokens 100_000_000
    python train.py --tag smoke --tokens 200_000 --eval-every 50   # sanity run

Every variant must share tokens/seed/tokenizer with its baseline — the looping
scheme is supposed to be the only difference.
"""

import argparse
import json
import math
import os
import pathlib
import time

import mlflow
import torch
import torch.nn.functional as F

import data
import diag
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


def objective(model, x, y, args, generator):
    """Целевая функция прогона. Обычный лосс — частный случай, когда всё выключено."""
    n_loops = None
    if args.loop_sampling == "uniform":
        n_loops = int(torch.randint(1, args.n_loops + 1, (1,), generator=generator).item())
    if not (args.deep_supervision or args.early_exit or args.progress_head):
        return model(x, y, n_loops=n_loops)[1]

    states, losses = model.walk(x, y, n_loops)
    loss = losses[-1].mean()
    if args.deep_supervision and len(losses) > 1:
        loss = loss + args.deep_supervision * torch.stack(losses[:-1]).mean()
    if model.exit_head is not None:
        p = model.halting(states)
        loss = loss + (p * torch.stack(losses, dim=-1)).sum(-1).mean()
        steps = torch.arange(p.shape[-1], device=p.device)
        prior = args.ponder_prior * (1 - args.ponder_prior) ** steps
        prior = prior / prior.sum()
        loss = loss + args.ponder_beta * F.kl_div(prior.log().expand_as(p), p,
                                                  reduction="batchmean")
    if model.progress is not None:
        gain = torch.stack([losses[t] - losses[t + 1] for t in range(len(losses) - 1)], dim=-1)
        pred = model.predicted_gain(states)[..., :gain.shape[-1]]
        loss = loss + args.progress_beta * F.mse_loss(pred, gain.detach())
    return loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--tokenizer", default="tokenizers/fineweb16k")
    p.add_argument("--tokens", type=int, default=100_000_000, help="training token budget")
    p.add_argument("--n-loops", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4, help="layers inside one loop block")
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--n-heads", type=int, default=6)
    p.add_argument("--n-kv-heads", type=int, default=3, help="Qwen3: вдвое меньше, чем голов")
    p.add_argument("--head-dim", type=int, default=128, help="Qwen3 задаёт его независимо от d_model")
    p.add_argument("--d-ff", type=int, default=1152)
    p.add_argument("--loop-scheme", default="stack", choices=["stack", "layer", "group"])
    p.add_argument("--group-size", type=int, default=2)
    p.add_argument("--input-injection", action="store_true",
                   help="подмешивать эмбеддинги на каждом лупе")
    p.add_argument("--loop-norm", action="store_true",
                   help="нормализовать h после каждого лупа")
    p.add_argument("--step-cond", action="store_true",
                   help="прибавлять эмбеддинг номера шага")
    p.add_argument("--deep-supervision", type=float, default=0.0,
                   help="вес лосса на промежуточных шагах: L_T + b * среднее(L_t). "
                        "Требует, чтобы луп улучшал предсказание, а не только доводил "
                        "до конца — анти-DEQ по смыслу")
    p.add_argument("--early-exit", action="store_true",
                   help="голова остановки, лосс PonderNet: sum p_t L_t + b KL(p||Geom)")
    p.add_argument("--ponder-beta", type=float, default=0.01)
    p.add_argument("--ponder-prior", type=float, default=0.3,
                   help="lambda геометрического приора: чем больше, тем раньше остановка")
    p.add_argument("--progress-head", action="store_true",
                   help="голова PALBERT: предсказывает, сколько лосса снимет следующий шаг")
    p.add_argument("--progress-beta", type=float, default=0.1)
    p.add_argument("--loop-sampling", choices=["fixed", "uniform"], default="fixed",
                   help="uniform — случайное число лупов на каждом батче, как в Huginn: "
                        "модель должна работать на любой глубине, а не только на своей")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="пересчитывать активации шага при бэкпропе; нужно при многих лупах")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--min-lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-batches", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=0,
                   help="сохранять чекпойнт каждые N шагов (0 = только лучший); "
                        "нужно, чтобы смотреть диагностику в динамике обучения")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--diag-every", type=int, default=0,
                   help="писать диагностику лупов в runs/<tag>/diag.jsonl каждые N шагов; "
                        "без неё видно только лосс, а не что происходит с состоянием")
    p.add_argument("--spectral-every", type=int, default=4,
                   help="считать спектральный радиус каждую N-ю запись диагностики; "
                        "он требует прямого режима дифференцирования и стоит дороже прочего")
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
        n_kv_heads=args.n_kv_heads,
        head_dim=args.head_dim,
        n_layers=args.n_layers,
        n_loops=args.n_loops,
        d_ff=args.d_ff,
        max_seq=args.seq_len,
        loop_scheme=args.loop_scheme,
        group_size=args.group_size,
        loop_norm=args.loop_norm,
        input_injection=args.input_injection,
        step_cond=args.step_cond,
        early_exit=args.early_exit,
        progress_head=args.progress_head,
        grad_checkpoint=args.grad_checkpoint,
    )
    model = LoopedLM(cfg).to(device)
    total, non_emb = model.n_params()
    print(f"device={device}  params: {total/1e6:.2f}M total / {non_emb/1e6:.2f}M non-embedding")
    if non_emb > 10e6:
        print("!! over the 10M budget — see NOTES for which reading we committed to")

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("looped-models")
    mlflow.start_run(run_name=args.tag)
    mlflow.log_params(vars(args))
    mlflow.log_params({"params_total": total, "params_non_embedding": non_emb})

    val, train = data.split(tok, args.seq_len, args.batch_size, args.val_batches, device)

    # веса нормализаций одномерные, и распад на них — не регуляризация, а сдвиг
    # масштаба активаций; в рецептах обучения языковых моделей их всегда исключают
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))
    tokens_per_step = args.batch_size * args.seq_len
    total_steps = args.tokens // tokens_per_step
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else None

    def write_history(best_loss):
        """Пишется после каждой оценки, а не в конце: иначе runs/<tag>/history.json
        всю дорогу хранит прошлый прогон под тем же тегом, и отчёт, собранный по ходу,
        показывает старую архитектуру как текущую."""
        (out / "history.json").write_text(json.dumps(
            {"config": vars(args), "params": {"total": total, "non_embedding": non_emb},
             "best_val_loss": best_loss if history else None,
             "best_val_ppl": math.exp(best_loss) if history else None,
             "history": history}, indent=2))

    dx, dy = val[0][0][:2], val[0][1][:2]  # фиксированный батч под диагностику
    sx = dx[:, :128]  # для спектрального радиуса хватает короткой последовательности
    diag_log = (out / "diag.jsonl").open("w")

    generator = torch.Generator().manual_seed(args.seed)
    best, history, t0 = float("inf"), [], time.time()
    for step, (x, y) in enumerate(train):
        if step >= total_steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total_steps, args.warmup, args.lr, args.min_lr)

        loss = objective(model, x, y, args, generator)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        by_block = diag.layer_grad_norms(model) if args.diag_every else {}
        opt.step()
        opt.zero_grad(set_to_none=True)

        if args.diag_every and step % args.diag_every == 0:
            rows = diag.loop_rows(model, dx, dy)
            opt.zero_grad(set_to_none=True)
            n = step // args.diag_every
            if args.spectral_every and n % args.spectral_every == 0:
                for row, rho in zip(rows, diag.spectral_by_step(model, sx)):
                    row["spectral_radius"] = rho
            diag_log.write(json.dumps({"step": step, "tokens": step * tokens_per_step,
                                       "train_loss": loss.item(), **by_block,
                                       "rows": rows}) + "\n")
            diag_log.flush()

        if step % args.log_every == 0:
            seen = step * tokens_per_step
            print(f"step {step:>6}/{total_steps}  tok {seen/1e6:6.1f}M  "
                  f"loss {loss.item():.4f}  {time.time()-t0:.0f}s")
            mlflow.log_metrics({"train_loss": loss.item(), "tokens": seen}, step=step)

        if (step and step % args.eval_every == 0) or step == total_steps - 1:
            vl = evaluate(model, val)
            history.append({"step": step, "tokens": step * tokens_per_step,
                            "val_loss": vl, "val_ppl": math.exp(vl)})
            print(f"  val loss {vl:.4f}  ppl {math.exp(vl):.2f}")
            mlflow.log_metrics({"val_loss": vl, "val_ppl": math.exp(vl)}, step=step)
            blob = {"cfg": cfg.__dict__, "model": model.state_dict(),
                    "step": step, "val_loss": vl, "tokenizer": args.tokenizer}
            if vl < best:
                best = vl
                torch.save(blob, out / "ckpt.pt")
            if args.save_every and step % args.save_every == 0:
                torch.save(blob, out / f"ckpt_step{step:06d}.pt")
            write_history(best)

    diag_log.close()
    best_ppl = math.exp(best) if history else None
    if history:
        mlflow.log_metrics({"best_val_loss": best, "best_val_ppl": best_ppl})
    mlflow.end_run()
    write_history(best)
    print(f"best val ppl {best_ppl} -> {out}")


if __name__ == "__main__":
    main()
    # torch и datasets оставляют живые треды, процесс виснет в exit() и серия
    # прогонов не двигается. Всё уже записано, выходим жёстко.
    os._exit(0)
