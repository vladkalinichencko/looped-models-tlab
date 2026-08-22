"""Diagnostics on real held-out tokens and the model's actual recurrent path."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from model import Config, LoopedLM


def token_kl(logp: torch.Tensor, next_logp: torch.Tensor) -> torch.Tensor:
    return (logp.exp() * (logp - next_logp)).sum(-1).clamp_min(0)


def _continue(model, state, embedded, cos, sin, steps):
    for _ in range(steps):
        state = model.recurrent_step(state, embedded, cos, sin)
    return state


@torch.no_grad()
def ablations(model: LoopedLM, x: torch.Tensor, states: list[torch.Tensor],
              targets: torch.Tensor):
    if len(states) < 3:
        return []
    embedded, cos, sin = model.encode(x)
    final_logp = model.decode(states[-1]).log_softmax(-1)
    rows = []
    total = len(states) - 1
    for step in range(1, total):
        variants = {
            "freeze": states[step],
            "reset": _continue(model, states[0], embedded, cos, sin, total - step),
            "zero": _continue(model, torch.zeros_like(states[step]), embedded, cos, sin, total - step),
            "skip": _continue(model, states[step - 1], embedded, cos, sin, total - step),
        }
        for name, state in variants.items():
            logp = model.decode(state).log_softmax(-1)
            loss = F.nll_loss(logp.reshape(-1, logp.shape[-1]), targets.reshape(-1))
            rows.append({"step": step, "intervention": name, "loss": loss.item(),
                         "kl_to_original": max(0.0, token_kl(logp, final_logp).mean().item())})
    return rows


def write_snapshot(model: LoopedLM, block: torch.Tensor, tok, out: Path,
                   train_step: int, device: str):
    was_training = model.training
    model.eval()
    x = block[None, :129].to(device)
    inputs, targets = x[:, :-1], x[:, 1:]
    steps = (max(model.cfg.mean_recurrence, model.cfg.backprop_last + 2)
             if model.cfg.backprop_last else model.cfg.mean_recurrence)
    states = list(model.states(inputs, steps))
    for state in states:
        if state.requires_grad:
            state.retain_grad()
    logits = [model.decode(state) for state in states]
    final_loss = F.cross_entropy(logits[-1].reshape(-1, logits[-1].shape[-1]), targets.reshape(-1))
    final_loss.backward()

    detached = [state.detach() for state in states]
    logp = [value.detach().log_softmax(-1) for value in logits]
    losses = [F.cross_entropy(value.reshape(-1, value.shape[-1]), targets.reshape(-1),
                              reduction="none").view_as(targets).detach() for value in logits]
    deltas = [detached[i + 1] - detached[i] for i in range(len(detached) - 1)]
    rows = []
    for step, delta in enumerate(deltas, 1):
        previous = deltas[step - 2] if step > 1 else None
        rows.append({
            "step": step,
            "state_norm": detached[step].norm(dim=-1).mean().item(),
            "delta_norm": delta.norm(dim=-1).mean().item(),
            "delta_cosine": None if previous is None else F.cosine_similarity(
                delta.flatten(0, 1), previous.flatten(0, 1), dim=-1).mean().item(),
            "token_kl": token_kl(logp[step - 1], logp[step]).mean().item(),
            "token_loss": losses[step].mean().item(),
            "state_grad": None if states[step - 1].grad is None else
                states[step - 1].grad.norm(dim=-1).mean().item(),
        })

    adjacent_kl = torch.stack([token_kl(logp[i], logp[i + 1]) for i in range(len(logp) - 1)])
    exits = torch.full(targets.shape, len(deltas), dtype=torch.long, device=targets.device)
    for step in range(len(deltas)):
        exits[(adjacent_kl[step] < 5e-4) & (exits == len(deltas))] = step + 1
    predictions = [value.argmax(-1) for value in logits]
    token_rows = [{
        "position": position,
        "input": tok.decode([int(inputs[0, position])]),
        "target": tok.decode([int(targets[0, position])]),
        "prediction": tok.decode([int(predictions[-1][0, position])]),
        "loss": losses[-1][0, position].item(),
        "exit_depth": int(exits[0, position]),
    } for position in range(inputs.shape[1])]

    snapshot_dir = out / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    tensor_path = snapshot_dir / f"step{train_step:06d}.pt"
    torch.save({
        "input_ids": inputs.cpu(), "targets": targets.cpu(),
        "states": [state.cpu() for state in detached],
        "deltas": [delta.cpu() for delta in deltas],
        "logits": [value.detach().half().cpu() for value in logits],
        "token_loss": [value.cpu() for value in losses],
        "adjacent_kl": adjacent_kl.cpu(), "exit_depth": exits.cpu(),
    }, tensor_path)
    summary = {
        "step": train_step, "tensor_path": str(tensor_path),
        "text": tok.decode(inputs[0].tolist()),
        "decoded_predictions": [tok.decode(row[0].tolist()) for row in predictions],
        "rows": rows, "tokens": token_rows,
        "ablations": ablations(model, inputs, detached, targets),
    }
    with (out / "diag.jsonl").open("a") as handle:
        handle.write(json.dumps(summary) + "\n")
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return summary


def fit_projection(snapshot: Path, out: Path = Path("runs/projection.pt")):
    raw = torch.load(snapshot, map_location="cpu", weights_only=False)
    values = torch.cat([state.float().flatten(0, 1) for state in raw["states"]])
    mean = values.mean(0)
    _, _, basis = torch.pca_lowrank(values - mean, q=2)
    torch.save({"mean": mean, "basis": basis, "source": str(snapshot)}, out)
    return out


def load(checkpoint: str | Path, device: str = "cpu"):
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    model = LoopedLM(Config(**blob["model_config"])).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, blob
