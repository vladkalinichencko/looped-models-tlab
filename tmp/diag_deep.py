"""Глубокая диагностика лупов: не «хорошо или плохо», а что именно происходит внутри.

Четыре среза, все на одной фиксированной последовательности:

1. Поток градиента. ||dL/dh_t|| на каждой границе лупа при бэкпропе от финального
   лосса. Показывает, затухает ли сигнал обучения через повторные применения блока.
2. Эволюция softmax. Для каждой позиции и каждого лупа — энтропия, вероятность
   топ-1 и момент, когда предсказание перестаёт меняться.
3. Один проход против двух из одного состояния. Второй проход продолжает движение
   в ту же сторону или разворачивает его.
4. Деформация пространства. Сетка точек в плоскости двух главных компонент,
   прогнанная несколько раз через MLP-подслой — видно, как блок гнёт пространство.
   Берём именно MLP: он поточечный, то есть честное отображение R^d -> R^d.
   Внимание так нарисовать нельзя, оно смешивает позиции и точкой не является.

    python tmp/diag_deep.py runs/loop4/ckpt.pt --loops 12
"""

import argparse
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import data  # noqa: E402
from model import Config, LoopedLM, rope_cache  # noqa: E402

SENTENCE = ("The capital of France is Paris, and the capital of Italy is Rome. "
            "Water boils at one hundred degrees celsius at sea level.")


def load(ckpt, device):
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = Config(**blob["cfg"])
    model = LoopedLM(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, cfg, blob["tokenizer"]


def apply_block_stack(model, h, cos, sin):
    for block in model.blocks:
        h = block(h, cos, sin)
    return h


def readout(model, h):
    return model.lm_head(model.norm(h))


# ---------- 1. поток градиента через лупы ----------
def gradient_flow(model, idx, targets, cos, sin, loops):
    h = model.embed(idx)
    marks = []
    for _ in range(loops):
        h = h.clone()
        h.retain_grad()
        marks.append(h)
        h = apply_block_stack(model, h, cos, sin)
    logits = readout(model, h)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    model.zero_grad(set_to_none=True)
    loss.backward()
    return [float(m.grad.norm()) for m in marks], float(loss)


# ---------- 2. эволюция предсказания ----------
@torch.no_grad()
def softmax_evolution(model, idx, cos, sin, loops):
    h = model.embed(idx)
    ent, top1p, top1id = [], [], []
    for _ in range(loops):
        h = apply_block_stack(model, h, cos, sin)
        logp = readout(model, h).log_softmax(-1)[0]
        p = logp.exp()
        ent.append(-(p * logp).sum(-1))
        best = p.max(-1)
        top1p.append(best.values)
        top1id.append(best.indices)
    return torch.stack(ent), torch.stack(top1p), torch.stack(top1id)


# ---------- 3. один проход против двух ----------
@torch.no_grad()
def one_vs_two(model, idx, targets, cos, sin, start_loops):
    h = model.embed(idx)
    for _ in range(start_loops):
        h = apply_block_stack(model, h, cos, sin)

    h1 = apply_block_stack(model, h, cos, sin)
    h2 = apply_block_stack(model, h1, cos, sin)

    d1, d2 = h1 - h, h2 - h1
    ce = lambda x: float(F.cross_entropy(
        readout(model, x).reshape(-1, model.cfg.vocab_size), targets.reshape(-1)))
    return {
        "start_loops": start_loops,
        "loss_0": ce(h), "loss_1": ce(h1), "loss_2": ce(h2),
        "step1": float(d1.norm()), "step2": float(d2.norm()),
        "cos_шагов": float(F.cosine_similarity(
            d1.reshape(-1, d1.shape[-1]), d2.reshape(-1, d2.shape[-1]), dim=-1).mean()),
    }


# ---------- 4. деформация пространства ----------
@torch.no_grad()
def space_warp(model, h_real, loops, grid=11, span=2.0):
    """Сетку в плоскости PC1-PC2 прогоняем через MLP-подслой несколько раз."""
    flat = h_real.reshape(-1, h_real.shape[-1]).float()
    mean = flat.mean(0)
    _, _, basis = torch.pca_lowrank(flat - mean, q=2)
    scale = (flat - mean).norm(dim=-1).median()

    lin = torch.linspace(-span, span, grid, device=flat.device) * scale
    gx, gy = torch.meshgrid(lin, lin, indexing="ij")
    pts = mean + gx.reshape(-1, 1) * basis[:, 0] + gy.reshape(-1, 1) * basis[:, 1]

    frames = [((pts - mean) @ basis).reshape(grid, grid, 2).cpu()]
    h = pts
    for _ in range(loops):
        for block in model.blocks:
            h = h + block.mlp(block.mlp_norm(h))
        frames.append(((h - mean) @ basis).reshape(grid, grid, 2).cpu())
    return frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--loops", type=int, default=12)
    p.add_argument("--device", default="cpu")
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    model, cfg, tok_name = load(args.ckpt, args.device)
    tok = data.tokenizer(tok_name)
    tag = args.tag or pathlib.Path(args.ckpt).parent.name

    ids = tok(SENTENCE, return_tensors="pt")["input_ids"][:, : cfg.max_seq].to(args.device)
    idx, targets = ids[:, :-1], ids[:, 1:]
    cos, sin = rope_cache(idx.shape[1], cfg.d_model // cfg.n_heads,
                          cfg.rope_theta, args.device)

    print(f"=== {tag}: обучен с {cfg.n_loops} лупами, последовательность {idx.shape[1]} токенов ===\n")

    grads, loss = gradient_flow(model, idx, targets, cos, sin, args.loops)
    print("1. Поток градиента (бэкпроп от финального лосса):")
    print(f"{'до лупа':>8} {'||dL/dh||':>12} {'во сколько раз к следующему':>28}")
    for i, g in enumerate(grads):
        ratio = f"{grads[i+1]/g:.3f}" if i + 1 < len(grads) and g > 0 else "—"
        print(f"{i:>8} {g:>12.3e} {ratio:>28}")

    ent, top1p, top1id = softmax_evolution(model, idx, cos, sin, args.loops)
    final = top1id[-1]
    settled = [(top1id[:, j] == final[j]).float().nonzero()[0].item() + 1
               if (top1id[:, j] == final[j]).any() else args.loops
               for j in range(top1id.shape[1])]
    print(f"\n2. Эволюция предсказания по {args.loops} лупам:")
    print(f"   энтропия: {ent[0].mean():.3f} -> {ent[-1].mean():.3f} нат")
    print(f"   вероятность топ-1: {top1p[0].mean():.3f} -> {top1p[-1].mean():.3f}")
    print(f"   медиана лупа, после которого топ-1 больше не меняется: "
          f"{sorted(settled)[len(settled)//2]}")
    print(f"   позиций, застывших уже после 1 лупа: "
          f"{sum(s <= 1 for s in settled)}/{len(settled)}")

    print("\n3. Один проход против двух из одного состояния:")
    print(f"{'старт':>6} {'лосс(h)':>9} {'лосс(+1)':>9} {'лосс(+2)':>9} "
          f"{'шаг1':>9} {'шаг2':>9} {'cos(шаг1,шаг2)':>15}")
    ovt = []
    for start in (1, 2, 4, 8):
        if start >= args.loops:
            continue
        r = one_vs_two(model, idx, targets, cos, sin, start)
        ovt.append(r)
        print(f"{start:>6} {r['loss_0']:>9.4f} {r['loss_1']:>9.4f} {r['loss_2']:>9.4f} "
              f"{r['step1']:>9.1f} {r['step2']:>9.1f} {r['cos_шагов']:>15.4f}")

    # ---- картинки ----
    with torch.no_grad():
        h_real = model.embed(idx)
        for _ in range(cfg.n_loops):
            h_real = apply_block_stack(model, h_real, cos, sin)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].semilogy(range(len(grads)), grads, marker="o")
    axes[0].set_xlabel("номер лупа"); axes[0].set_ylabel("||dL/dh||")
    axes[0].set_title("поток градиента через лупы"); axes[0].grid(alpha=.3)

    im = axes[1].imshow(ent.cpu(), aspect="auto", origin="lower", cmap="viridis")
    axes[1].set_xlabel("позиция в тексте"); axes[1].set_ylabel("луп")
    axes[1].set_title("энтропия предсказания"); fig.colorbar(im, ax=axes[1])

    axes[2].plot(range(1, args.loops + 1), top1p.mean(1).cpu(), marker="o")
    axes[2].set_xlabel("луп"); axes[2].set_ylabel("средняя вероятность топ-1")
    axes[2].set_title("уверенность по лупам"); axes[2].grid(alpha=.3)
    fig.suptitle(f"{tag}: обучен с {cfg.n_loops} лупами")
    fig.tight_layout(); fig.savefig(f"tmp/deep_{tag}.png", dpi=140)

    frames = space_warp(model, h_real, loops=4)
    fig, axes = plt.subplots(1, len(frames), figsize=(3.2 * len(frames), 3.4))
    for k, (ax, fr) in enumerate(zip(axes, frames)):
        for i in range(fr.shape[0]):
            ax.plot(fr[i, :, 0], fr[i, :, 1], lw=.7, color="tab:blue", alpha=.8)
            ax.plot(fr[:, i, 0], fr[:, i, 1], lw=.7, color="tab:blue", alpha=.8)
        ax.set_title(f"после {k} лупов" if k else "исходная сетка")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{tag}: как MLP гнёт пространство (плоскость PC1–PC2)")
    fig.tight_layout(); fig.savefig(f"tmp/warp_{tag}.png", dpi=140)

    out = pathlib.Path("tmp") / f"deep_{tag}.json"
    out.write_text(json.dumps({
        "tag": tag, "trained_loops": cfg.n_loops, "final_loss": loss,
        "grad_norms": grads, "entropy": ent.mean(1).tolist(),
        "top1_prob": top1p.mean(1).tolist(), "settled_loop": settled,
        "one_vs_two": ovt}, indent=2))
    print(f"\n-> tmp/deep_{tag}.png, tmp/warp_{tag}.png, {out}")


if __name__ == "__main__":
    main()
