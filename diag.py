"""What the state does between loops, as numbers.

Perplexity only says "worse". These say why: is the state converging to a fixed
point, does the step move it where the loss wants, does the rank collapse. Every
metric here is a pure function of (model, batch), so train.py logs the same numbers
during training that the CLI reports for a finished checkpoint.

    python diag.py runs/loop4/ckpt.pt --loops 16
"""

import argparse
import json
import os
import pathlib

import torch
import torch.nn.functional as F

import data
from model import Config, LoopedLM


def effective_rank(x):
    """Participation ratio of the singular values: how many directions are in use.

    MPS has no SVD, so this one op runs on CPU.
    """
    flat = x.reshape(-1, x.shape[-1]).float().cpu()
    flat = flat - flat.mean(0, keepdim=True)
    sv = torch.linalg.svdvals(flat)
    p = sv / sv.sum()
    return float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))


def intrinsic_dim(x, n=1024, seed=0):
    """TwoNN estimate: the slope of -log(1-F) against log(r2/r1) is the dimension.

    Effective rank counts linear directions, this one counts the dimension of the
    curved sheet the states actually lie on — the two disagree exactly when the
    manifold is not a subspace.
    """
    flat = x.reshape(-1, x.shape[-1]).float().cpu()
    idx = torch.randperm(flat.shape[0], generator=torch.Generator().manual_seed(seed))[:n]
    d = torch.cdist(flat[idx], flat[idx])
    r, _ = d.topk(2, dim=1, largest=False, sorted=True)
    mu = (d.topk(3, dim=1, largest=False).values[:, 2] / r[:, 1].clamp_min(1e-9))
    mu = mu[mu > 1].sort().values
    f = torch.arange(1, len(mu) + 1).float() / (len(mu) + 1)
    lx, ly = mu.log(), -(1 - f).log()
    return float((lx @ ly) / (lx @ lx).clamp_min(1e-12))


def spectral_radius(step, h, iters=20, seed=0):
    """Power iteration on the Jacobian of one loop step at h.

    Below 1 the map is a contraction: Banach guarantees a unique fixed point and
    geometric convergence, so extra loops provably add nothing new. Above 1 the
    state keeps moving. This is the number that decides whether "more loops" is
    even possible, and it costs one JVP per iteration, not a full Jacobian.
    """
    v = torch.randn(h.shape, generator=torch.Generator().manual_seed(seed)).to(h)
    v = v / v.norm()
    lam = 0.0
    for _ in range(iters):
        _, jv = torch.func.jvp(step, (h,), (v,))
        lam = float(jv.norm())
        v = jv / max(lam, 1e-12)
    return lam


def loop_rows(model, x, y, n_loops=None, with_spectral=False):
    """One row per loop step: convergence, usefulness, rank, gradient, prediction."""
    states = []
    for h in model.trace(x, n_loops):
        h.retain_grad()
        states.append(h)
    logits = model.head(states[-1])
    F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).backward()
    through = [s.grad.norm().item() for s in states]  # реально текущий назад через лупы

    states = [s.detach() for s in states]
    p_final = model.head(states[-1]).log_softmax(-1)
    top_final = p_final.argmax(-1)
    prev = None
    rows = []
    for t in range(len(states) - 1):
        h0, h1 = states[t], states[t + 1]
        dh = h1 - h0

        exit_h = h0.clone().requires_grad_(True)
        exit_logits = model.head(exit_h)
        loss_exit = F.cross_entropy(exit_logits.reshape(-1, exit_logits.size(-1)), y.reshape(-1))
        (g,) = torch.autograd.grad(loss_exit, exit_h)

        with torch.no_grad():
            logp = model.head(h1).log_softmax(-1)
            flat_dh = dh.reshape(-1, dh.shape[-1])
            rows.append({
                "step": t + 1,
                "h_norm": float(h1.norm(dim=-1).mean()),
                "dh_norm": float(dh.norm(dim=-1).mean()),
                "rel_step": float(dh.norm() / h0.norm()),
                "cos_prev": None if prev is None else float(
                    F.cosine_similarity(flat_dh, prev, dim=-1).mean()),
                "cos_useful": float(F.cosine_similarity(
                    flat_dh, -g.reshape(-1, g.shape[-1]), dim=-1).mean()),
                "eff_rank": effective_rank(h1),
                "intrinsic_dim": intrinsic_dim(h1),
                "grad_in": through[t],
                "grad_out": through[t + 1],
                "grad_exit": float(g.norm()),
                "kl_to_final": float(F.kl_div(p_final, logp, log_target=True,
                                              reduction="batchmean")),
                "top1_changed": float((logp.argmax(-1) != top_final).float().mean()),
                "entropy": float(-(logp.exp() * logp).sum(-1).mean()),
                "top1_prob": float(logp.max(-1).values.exp().mean()),
                "loss": float(F.cross_entropy(
                    model.head(h1).reshape(-1, logits.size(-1)), y.reshape(-1))),
            })
        prev = flat_dh

    if with_spectral:
        plan = model.plan(n_loops)
        for t, row in enumerate(rows):
            blocks = [model.blocks[i] for i in plan[t]]
            cos, sin = _rope(model, x)

            def step(h, blocks=blocks):
                for b in blocks:
                    h = b(h, cos, sin)
                return h

            row["spectral_radius"] = spectral_radius(step, states[t])
    return rows


def layer_grad_norms(model):
    """Gradient norm per block, for the same question one layer down: which of the
    shared blocks still gets signal after the loop stack has been unrolled."""
    return {f"grad_block{i}": sum(float(p.grad.norm()) ** 2
                                  for p in b.parameters() if p.grad is not None) ** 0.5
            for i, b in enumerate(model.blocks)}


def pca_traj(states, n_tokens=16, k=2):
    """Trajectory of the first tokens in the plane of the first two components."""
    stack = torch.stack([s[0, :n_tokens].float().cpu() for s in states])
    flat = stack.reshape(-1, stack.shape[-1])
    _, _, v = torch.pca_lowrank(flat - flat.mean(0, keepdim=True), q=k)
    return (stack @ v).tolist()


def _rope(model, x):
    from model import rope_cache
    cfg = model.cfg
    return rope_cache(x.shape[1], cfg.d_model // cfg.n_heads, cfg.rope_theta, x.device)


def load(ckpt, device="cpu"):
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    model = LoopedLM(Config(**blob["cfg"])).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, blob


def val_batches(blob, cfg, batch_size, batches, device):
    """Cached, otherwise every diagnostic run re-streams FineWeb over the network."""
    cache = pathlib.Path("datasets") / f"val_{cfg.max_seq}_{batch_size}_{batches}.pt"
    if cache.exists():
        return [(x.to(device), y.to(device)) for x, y in torch.load(cache)]
    val, _ = data.split(data.tokenizer(blob["tokenizer"]), cfg.max_seq, batch_size,
                        batches, device)
    cache.parent.mkdir(exist_ok=True)
    torch.save([(x.cpu(), y.cpu()) for x, y in val], cache)
    return val


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--loops", type=int, default=None, help="сколько шагов прогнать")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--batches", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    model, blob = load(args.ckpt, args.device)
    val = val_batches(blob, model.cfg, args.batch_size, args.batches, args.device)

    rows = loop_rows(model, *val[0], n_loops=args.loops, with_spectral=True)
    print(f"{'шаг':>4} {'‖h‖':>9} {'шаг/‖h‖':>9} {'cos пред':>9} {'cos польз':>10} "
          f"{'ранг':>7} {'ρ(J)':>7} {'KL фин':>8} {'лосс':>8}")
    for r in rows:
        print(f"{r['step']:>4} {r['h_norm']:>9.2f} {r['rel_step']:>9.4f} "
              f"{-9.0 if r['cos_prev'] is None else r['cos_prev']:>9.4f} "
              f"{r['cos_useful']:>10.4f} {r['eff_rank']:>7.1f} "
              f"{r['spectral_radius']:>7.3f} {r['kl_to_final']:>8.4f} {r['loss']:>8.4f}")

    states = [h.detach() for h in model.trace(val[0][0], args.loops)]
    out = pathlib.Path(args.out or pathlib.Path(args.ckpt).parent / "diag.json")
    out.write_text(json.dumps({"ckpt": args.ckpt, "cfg": model.cfg.__dict__,
                               "rows": rows, "traj": pca_traj(states)}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
    # torch и datasets оставляют живые треды, процесс виснет в exit() и серия по
    # чекпойнтам не двигается. Всё уже записано, выходим жёстко.
    os._exit(0)
