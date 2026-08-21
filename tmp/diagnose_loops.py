"""Что происходит с представлением от лупа к лупу.

Перплексия говорит только «стало хуже». Здесь смотрим, почему: сходится ли состояние
к неподвижной точке, двигают ли лупы его в полезную сторону, не схлопывается ли ранг.

Ключевая метрика — cos(dh, -dL/dh): куда луп сдвинул состояние против того, куда его
надо было сдвинуть, чтобы уменьшить лосс. Положительно — луп делает работу.
Около нуля — крутит вхолостую.

    python tmp/diagnose_loops.py runs/dense1/ckpt.pt --loops 12
    python tmp/diagnose_loops.py runs/loop4/ckpt.pt --loops 12 --tag loop4
"""

import argparse
import json
import os
import pathlib
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import data  # noqa: E402
from model import Config, LoopedLM, rope_cache  # noqa: E402


def loop_states(model, idx, n_loops):
    """Состояние h после каждого лупа. h[0] — сразу после эмбеддингов."""
    cfg = model.cfg
    cos, sin = rope_cache(idx.shape[1], cfg.d_model // cfg.n_heads, cfg.rope_theta, idx.device)
    h = model.embed(idx)
    states = [h]
    for _ in range(n_loops):
        for block in model.blocks:
            h = block(h, cos, sin)
        states.append(h)
    return states


def effective_rank(x):
    """Participation ratio сингулярных чисел: сколько направлений реально заняты.

    SVD на MPS не реализован, поэтому эту одну операцию считаем на CPU.
    """
    flat = x.reshape(-1, x.shape[-1]).float().cpu()
    flat = flat - flat.mean(0, keepdim=True)
    sv = torch.linalg.svdvals(flat)
    p = sv / sv.sum()
    return float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))


def logits_from(model, h):
    return model.lm_head(model.norm(h))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--loops", type=int, default=12, help="сколько лупов прогнать")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--batches", type=int, default=4, help="по скольким батчам усреднять")
    p.add_argument("--device", default="cpu")
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    blob = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = Config(**blob["cfg"])
    model = LoopedLM(cfg).to(args.device)
    model.load_state_dict(blob["model"])
    model.eval()

    # валидационные батчи кэшируем: без этого каждый запуск заново тянет FineWeb
    # по сети, и на серии чекпойнтов диагностика просто зависает
    cache = pathlib.Path("datasets") / f"val_{cfg.max_seq}_{args.batch_size}_{args.batches}.pt"
    if cache.exists():
        val = [(x.to(args.device), y.to(args.device)) for x, y in torch.load(cache)]
    else:
        tok = data.tokenizer(blob["tokenizer"])
        val, _ = data.split(tok, cfg.max_seq, args.batch_size, args.batches, args.device)
        cache.parent.mkdir(exist_ok=True)
        torch.save([(x.cpu(), y.cpu()) for x, y in val], cache)
        print(f"валидационные батчи закэшированы: {cache}")

    rows = []
    traj = []  # траектория первых токенов в PCA, только с первого батча
    for bi, (x, y) in enumerate(val):
        states = loop_states(model, x, args.loops)

        # градиент лосса по каждому промежуточному состоянию
        grads = []
        for h in states[:-1]:
            h = h.detach().requires_grad_(True)
            logits = logits_from(model, h)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            (g,) = torch.autograd.grad(loss, h)
            grads.append(g)

        with torch.no_grad():
            p_final = logits_from(model, states[-1]).log_softmax(-1)
            for t in range(args.loops):
                h0, h1, g = states[t], states[t + 1], grads[t]
                dh = h1 - h0
                useful = F.cosine_similarity(
                    dh.reshape(-1, dh.shape[-1]), -g.reshape(-1, g.shape[-1]), dim=-1
                ).mean()

                logits_t = logits_from(model, h1)
                logp_t = logits_t.log_softmax(-1)
                kl = F.kl_div(p_final, logp_t, log_target=True, reduction="batchmean")
                loss_t = F.cross_entropy(logits_t.reshape(-1, logits_t.size(-1)),
                                         y.reshape(-1))

                row = {
                    "loop": t + 1,
                    "h_norm": float(h1.norm(dim=-1).mean().item()),
                    "rel_step": float((dh.norm() / h0.norm()).item()),
                    "cos_useful": float(useful.item()),
                    "eff_rank": effective_rank(h1),
                    "grad_norm": float(g.norm().item()),
                    "kl_to_final": float(kl.item()),
                    "loss": float(loss_t.item()),
                    "entropy": float(-(logp_t.exp() * logp_t).sum(-1).mean().item()),
                }
                if bi == 0:
                    rows.append(row)
                else:
                    for k, v in row.items():
                        if k != "loop":
                            rows[t][k] += v

            if bi == 0:
                traj = [s[0, :16].detach().cpu() for s in states]

    for row in rows:
        for k in row:
            if k != "loop":
                row[k] /= len(val)

    print(f"{'луп':>4} {'‖h‖':>9} {'шаг/‖h‖':>9} {'cos полезн':>11} {'эфф.ранг':>9} "
          f"{'KL до фин':>10} {'лосс':>8}")
    for r in rows:
        print(f"{r['loop']:>4} {r['h_norm']:>9.2f} {r['rel_step']:>9.4f} "
              f"{r['cos_useful']:>11.4f} {r['eff_rank']:>9.1f} "
              f"{r['kl_to_final']:>10.4f} {r['loss']:>8.4f}")

    tag = args.tag or pathlib.Path(args.ckpt).parent.name
    out = pathlib.Path("tmp") / f"diag_{tag}.json"
    out.write_text(json.dumps({"ckpt": args.ckpt, "trained_loops": cfg.n_loops,
                               "rows": rows}, indent=2))

    # траектория состояния в первых двух главных компонентах
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        stack = torch.stack(traj)  # (loops+1, tokens, d)
        flat = stack.reshape(-1, stack.shape[-1]).float()
        flat = flat - flat.mean(0, keepdim=True)
        _, _, v = torch.pca_lowrank(flat, q=2)
        proj = (stack.float() @ v)  # (loops+1, tokens, 2)

        fig, ax = plt.subplots(figsize=(6, 5))
        for token in range(proj.shape[1]):
            ax.plot(proj[:, token, 0], proj[:, token, 1], marker="o", ms=2, alpha=0.6)
            ax.scatter(proj[0, token, 0], proj[0, token, 1], c="k", s=12, zorder=3)
        ax.set_title(f"{tag}: траектория h по лупам (чёрное — старт)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.tight_layout()
        fig.savefig(f"tmp/traj_{tag}.png", dpi=140)
        print(f"-> tmp/traj_{tag}.png")
    except Exception as exc:
        print(f"график не построился: {exc}")

    print(f"-> {out}")


if __name__ == "__main__":
    main()
    # torch/datasets оставляют живые треды, и процесс виснет в exit(), из-за чего
    # серия по чекпойнтам не двигается. Выходим жёстко, всё уже записано.
    os._exit(0)
