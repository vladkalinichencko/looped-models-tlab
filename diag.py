"""Diagnostics on real held-out tokens and the model's actual recurrent path."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from torch import nn

from model import Config, LoopedLM

def token_kl(logp: torch.Tensor, next_logp: torch.Tensor) -> torch.Tensor:
    return (logp.exp() * (logp - next_logp)).sum(-1).clamp_min(0)

def token_js(logp: torch.Tensor, next_logp: torch.Tensor) -> torch.Tensor:
    middle = torch.logaddexp(logp, next_logp) - torch.log(torch.tensor(2.0, device=logp.device))
    return 0.5 * (token_kl(logp, middle) + token_kl(next_logp, middle))

def _continue(model, state, embedded, cos, sin, steps, start_step=0):
    routing = (model.controller_routing(embedded, start_step + steps)
               if getattr(model, "controller_head", None) is not None else None)
    for step in range(start_step, start_step + steps):
        weights = None if routing is None else routing[step][1]
        state = model.recurrent_step(state, embedded, cos, sin, routing=weights)
    return state

@torch.no_grad()
def ablations(model: nn.Module, x: torch.Tensor, states: list[torch.Tensor],
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
            "reset": _continue(model, states[0], embedded, cos, sin, total - step, step),
            "zero": _continue(model, torch.zeros_like(states[step]), embedded, cos, sin,
                              total - step, step),
            "skip": _continue(model, states[step - 1], embedded, cos, sin, total - step, step),
        }
        for name, state in variants.items():
            logp = model.decode(state).log_softmax(-1)
            loss = F.nll_loss(logp.reshape(-1, logp.shape[-1]), targets.reshape(-1))
            rows.append({"step": step, "intervention": name, "loss": loss.item(),
                         "kl_to_original": max(0.0, token_kl(logp, final_logp).mean().item())})
    return rows

def write_snapshot(model: nn.Module, block: torch.Tensor, tok, out: Path,
                   train_step: int, device: str, projection: Path | None = None,
                   snapshot_name: str | None = None):
    was_training = model.training
    model.eval()
    x = block[None, :129].to(device)
    inputs, targets = x[:, :-1], x[:, 1:]
    steps = (max(model.cfg.mean_recurrence, model.cfg.backprop_last + 2)
             if model.cfg.backprop_last else model.cfg.mean_recurrence)
    layer_trace = []
    states = list(model.states(inputs, steps, trace=layer_trace))
    for state in states + [value for _, _, value in layer_trace]:
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
    adapter_components = None
    if getattr(model, "adapter", None) is not None:
        embedded = model.encode(inputs)[0].detach()
        state_weight, input_weight = model.adapter.weight.detach().split(model.cfg.d_model, 1)
        adapter_components = [(F.linear(state, state_weight), F.linear(embedded, input_weight))
                              for state in detached[:-1]]
    rows = []
    for step, delta in enumerate(deltas, 1):
        previous = deltas[step - 2] if step > 1 else None
        probs = logp[step].exp()
        top2 = probs.topk(2, dim=-1).values
        grad = states[step - 1].grad
        entropy = -(probs * logp[step]).sum(-1)
        kl = token_kl(logp[step - 1], logp[step])
        state_cosine = F.cosine_similarity(detached[step - 1], detached[step], dim=-1)
        components = None if adapter_components is None else adapter_components[step - 1]
        source = detached[step - 1]
        radial = ((delta * source).sum(-1, keepdim=True) /
                  source.square().sum(-1, keepdim=True).clamp_min(1e-12)) * source
        tangent = delta - radial
        rows.append({
            "step": step,
            "state_norm": detached[step].norm(dim=-1).mean().item(),
            "delta_norm": delta.norm(dim=-1).mean().item(),
            "radial_update": radial.norm(dim=-1).mean().item(),
            "tangent_update": tangent.norm(dim=-1).mean().item(),
            "relative_norm_drift": ((detached[step].square().sum(-1) - source.square().sum(-1)) /
                                    source.square().sum(-1).clamp_min(1e-12)).mean().item(),
            "state_cosine": state_cosine.mean().item(),
            "delta_cosine": None if previous is None else F.cosine_similarity(
                delta.flatten(0, 1), previous.flatten(0, 1), dim=-1).mean().item(),
            "token_kl": kl.mean().item(),
            "token_js": token_js(logp[step - 1], logp[step]).mean().item(),
            "entropy": entropy.mean().item(),
            "margin": (top2[..., 0] - top2[..., 1]).mean().item(),
            "top1_flip": (logp[step - 1].argmax(-1) != logp[step].argmax(-1)).float().mean().item(),
            "logit_change_rms": (logits[step].detach() - logits[step - 1].detach()).float().pow(2).mean().sqrt().item(),
            "token_loss": losses[step].mean().item(),
            "state_grad": None if grad is None else grad.norm(dim=-1).mean().item(),
            "grad_delta_cosine": None if grad is None else F.cosine_similarity(
                grad.flatten(0, 1), delta.flatten(0, 1), dim=-1).mean().item(),
            "state_norm_by_token": detached[step].norm(dim=-1)[0].cpu().tolist(),
            "delta_norm_by_token": delta.norm(dim=-1)[0].cpu().tolist(),
            "radial_update_by_token": radial.norm(dim=-1)[0].cpu().tolist(),
            "tangent_update_by_token": tangent.norm(dim=-1)[0].cpu().tolist(),
            "loss_by_token": losses[step][0].cpu().tolist(),
            "entropy_by_token": entropy[0].cpu().tolist(),
            "kl_by_token": kl[0].cpu().tolist(),
            "state_grad_by_token": None if grad is None else grad.norm(dim=-1)[0].cpu().tolist(),
            "state_cosine_by_token": state_cosine[0].cpu().tolist(),
            "adapter_state_by_token": None if components is None else
                components[0].norm(dim=-1)[0].cpu().tolist(),
            "adapter_input_by_token": None if components is None else
                components[1].norm(dim=-1)[0].cpu().tolist(),
        })

    adjacent_kl = torch.stack([token_kl(logp[i], logp[i + 1]) for i in range(len(logp) - 1)])
    predictions = [value.argmax(-1) for value in logits]
    token_rows = []
    for position in range(inputs.shape[1]):
        per_step = []
        for index, values in enumerate(logp):
            probs = values[0, position].exp()
            top = probs.topk(5)
            per_step.append({
                "step": index,
                "prediction": tok.decode([int(top.indices[0])]),
                "topk": [{"token": tok.decode([int(token)]), "p": float(probability)}
                         for token, probability in zip(top.indices, top.values)],
                "confidence": float(top.values[0]),
                "top5_mass": float(top.values.sum()),
                "loss": losses[index][0, position].item(),
                "entropy": float(-(probs * values[0, position]).sum()),
                "margin": float(top.values[0] - top.values[1]),
                "kl": None if index == 0 else float(token_kl(
                    logp[index - 1][0, position], values[0, position])),
                "js": None if index == 0 else float(token_js(
                    logp[index - 1][0, position], values[0, position])),
                "flip": None if index == 0 else bool(
                    logp[index - 1][0, position].argmax() != values[0, position].argmax()),
            })
        token_rows.append({
            "position": position,
            "input": tok.decode([int(inputs[0, position])]),
            "target": tok.decode([int(targets[0, position])]),
            "prediction": tok.decode([int(predictions[-1][0, position])]),
            "loss": losses[-1][0, position].item(),
            "exit_depth": None,
            "steps": per_step,
        })

    layer_rows = []
    previous = states[0].detach()
    phase_previous = {}
    with torch.no_grad():
        for loop, name, value in layer_trace:
            hidden = value.detach()
            prior_phase = phase_previous.get(name)
            phase_delta = None if prior_phase is None else hidden - prior_phase
            layer_logp = model.decode(hidden).log_softmax(-1)
            layer_loss = F.nll_loss(layer_logp.reshape(-1, layer_logp.shape[-1]),
                                    targets.reshape(-1), reduction="none").view_as(targets)
            probs = layer_logp.exp()
            entropy = -(probs * layer_logp).sum(-1)
            gradient = None if value.grad is None else value.grad.detach().norm(dim=-1)[0]
            layer_rows.append({
                "loop": loop, "name": name,
                "state_norm": hidden.norm(dim=-1).mean().item(),
                "delta_norm": (hidden - previous).norm(dim=-1).mean().item(),
                "loss": layer_loss.mean().item(),
                "entropy": entropy.mean().item(),
                "state_grad": None if gradient is None else gradient.mean().item(),
                "state_norm_by_token": hidden.norm(dim=-1)[0].cpu().tolist(),
                "delta_norm_by_token": (hidden - previous).norm(dim=-1)[0].cpu().tolist(),
                "loss_by_token": layer_loss[0].cpu().tolist(),
                "entropy_by_token": entropy[0].cpu().tolist(),
                "state_grad_by_token": None if gradient is None else gradient.cpu().tolist(),
                "phase_relative_by_token": None if phase_delta is None else
                    (phase_delta.norm(dim=-1) / prior_phase.norm(dim=-1).clamp_min(1e-12))[0]
                    .cpu().tolist(),
                "phase_delta_norm_by_token": None if phase_delta is None else
                    phase_delta.norm(dim=-1)[0].cpu().tolist(),
            })
            previous = hidden
            phase_previous[name] = hidden

    snapshot_dir = out / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    tensor_path = snapshot_dir / (f"{snapshot_name}.pt" if snapshot_name else
                                  f"step{train_step:06d}.pt")
    tensors = {
        "input_ids": inputs.cpu(), "targets": targets.cpu(),
        "states": [state.cpu() for state in detached],
        "deltas": [delta.cpu() for delta in deltas],
        "logits": [value.detach().half().cpu() for value in logits],
        "token_loss": [value.cpu() for value in losses],
        "adjacent_kl": adjacent_kl.cpu(),
        "layer_states": [value.detach().cpu() for _, _, value in layer_trace],
    }
    summary = {
        "step": train_step, "device": device, "tensor_path": str(tensor_path),
        "text": tok.decode(inputs[0].tolist()),
        "decoded_predictions": [tok.decode(row[0].tolist()) for row in predictions],
        "rows": rows, "layers": layer_rows, "tokens": token_rows,
        "ablations": ablations(model, inputs, detached, targets),
    }
    directional = []
    for state, delta in zip(detached[:-1], deltas):
        effect, fisher_length = _step_sensitivity(model, state, delta)
        directional.append({"rms": effect.mean().item(),
                            "rms_by_token": effect[0].cpu().tolist(),
                            "hidden_fisher_length": fisher_length.mean().item(),
                            "hidden_fisher_length_by_token": fisher_length[0].cpu().tolist()})
    summary["logit_directional_effect"] = directional
    active_rows = []
    for step, signal in enumerate(directional):
        entropy = -(logp[step].exp() * logp[step]).sum(-1)[0].cpu()
        js = token_js(logp[step], logp[step + 1])[0].cpu()
        improvement = (losses[step] - losses[step + 1])[0].cpu()
        scores = {
            "entropy": entropy,
            "js": js,
            "logit_sensitivity": torch.tensor(signal["rms_by_token"]),
            "hidden_fisher_length": torch.tensor(signal["hidden_fisher_length_by_token"]),
        }
        centered_target = improvement - improvement.mean()
        correlations = {}
        for name, score in scores.items():
            centered = score - score.mean()
            denominator = centered.norm() * centered_target.norm()
            correlations[name] = float((centered @ centered_target) /
                                       denominator.clamp_min(1e-12))
        active_rows.append({"step": step + 1, "loss_improvement": improvement.tolist(),
                            "correlations": correlations})
    summary["active_learning"] = active_rows
    if getattr(model, "controller_head", None) is not None:
        embedded, cos, sin = model.encode(inputs)
        routing = model.controller_routing(embedded, steps)
        controller_rows = []
        with torch.no_grad():
            for step, ((routing_logits, weights), state) in enumerate(zip(routing, detached), 1):
                proposals = torch.stack([block(state, cos, sin) for block in model.core], 2)
                normalized = F.normalize(proposals, dim=-1)
                similarity = normalized @ normalized.transpose(-1, -2)
                branch_logits = torch.stack([model.decode(proposals[:, :, k]) for k in range(4)], 2)
                branch_loss = F.cross_entropy(
                    branch_logits.reshape(-1, branch_logits.shape[-1]),
                    targets[..., None].expand(-1, -1, 4).reshape(-1), reduction="none",
                ).view(*targets.shape, 4)
                oracle = branch_loss.argmin(-1)
                controller_rows.append({
                    "step": step,
                    "weights": weights.mean((0, 1)).cpu().tolist(),
                    "entropy": (-(weights * weights.clamp_min(1e-12).log()).sum(-1)).mean().item(),
                    "oracle_frequency": F.one_hot(oracle, 4).float().mean((0, 1)).cpu().tolist(),
                    "oracle_accuracy": (routing_logits.argmax(-1) == oracle).float().mean().item(),
                    "branch_loss": branch_loss.mean((0, 1)).cpu().tolist(),
                    "pairwise_cosine": similarity.mean((0, 1)).cpu().tolist(),
                })
        summary["controller"] = controller_rows
    if projection is not None:
        analysis, extra = analyze(model, inputs, targets, detached, projection)
        summary.update(analysis)
        tensors.update(extra)
        reference = torch.load(projection, map_location=device, weights_only=False)
        for row, (_, _, value) in zip(layer_rows, layer_trace):
            projected = ((value.detach()[0] - reference["mean"]) @
                         reference["basis"][:, :3]).cpu()
            row["projection"] = projected[15, :2].tolist()
            row["projection_all"] = projected.tolist()
    torch.save(tensors, tensor_path)
    if snapshot_name:
        (out / f"{snapshot_name}.json").write_text(json.dumps(summary))
    else:
        with (out / "diag.jsonl").open("a") as handle:
            handle.write(json.dumps(summary) + "\n")
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return summary

def fit_projection(snapshot: Path, out: Path = Path("runs/projection.pt")):
    raw = torch.load(snapshot, map_location="cpu", weights_only=False)
    values = torch.cat([state.float().flatten(0, 1) for state in raw["states"]])
    mean = values.mean(0)
    _, singular, vh = torch.linalg.svd(values - mean, full_matrices=False)
    eigenvalues = singular.square()
    effective_rank = float(eigenvalues.sum().square() /
                           eigenvalues.square().sum().clamp_min(1e-12))
    basis = vh.T.contiguous()
    projected = (values - mean) @ basis[:, :2]
    torch.save({"mean": mean, "basis": basis, "singular": singular,
                "effective_rank": effective_rank,
                "limits": torch.stack([projected.amin(0), projected.amax(0)]),
                "reference": ((values - mean) @ basis[:, :3]),
                "source": str(snapshot)}, out)
    return out

def _jvp(function, point, direction):
    with sdpa_kernel(SDPBackend.MATH):
        return torch.func.jvp(function, (point,), (direction,))[1]

def _quantize(matrix: torch.Tensor):
    scale = matrix.abs().max().clamp_min(1e-12)
    return (matrix / scale * 127).round().to(torch.int8), float(scale)

def _jacobian(model: nn.Module, inputs: torch.Tensor, steps: int,
              initial_state: torch.Tensor):
    inputs = inputs[:, :16]
    embedded, cos, sin = model.encode(inputs)
    states = list(model.states(inputs, steps, initial_state=initial_state[:, :16]))
    state = states[-2].detach()

    routing = (model.controller_routing(embedded, steps)[-1][1]
               if getattr(model, "controller_head", None) is not None else None)

    def update(value):
        return model.recurrent_step(value, embedded, cos, sin, routing=routing)

    position = state.shape[1] - 1
    token = state[0, position].detach().requires_grad_(True)

    def local(value):
        replaced = state.clone()
        replaced[0, position] = value
        return update(replaced)[0, position]

    try:
        block = torch.autograd.functional.jacobian(local, token, vectorize=True)
    except (RuntimeError, NotImplementedError):
        block = torch.autograd.functional.jacobian(local, token, vectorize=False)
    block_cpu = block.detach().float().cpu()
    singular = torch.linalg.svdvals(block_cpu)
    eigen = torch.linalg.eigvals(block_cpu)
    quantized, scale = _quantize(block_cpu)

    generator = torch.Generator().manual_seed(0)
    sensitivity = torch.zeros(state.shape[1], state.shape[1])
    for source in range(state.shape[1]):
        direction = torch.zeros_like(state)
        direction[0, source] = torch.randn(state.shape[-1], generator=generator).to(state.device)
        direction /= direction.norm()
        response = _jvp(update, state, direction)[0].norm(dim=-1).cpu()
        sensitivity[:, source] = response

    direction = torch.randn(state.shape, generator=generator).to(state.device)
    direction /= direction.norm()
    for _ in range(4):
        response = _jvp(update, state, direction)
        point = state.detach().requires_grad_(True)
        transpose = torch.autograd.grad(update(point), point, response)[0]
        direction = transpose.detach() / transpose.norm().clamp_min(1e-12)
    response = _jvp(update, state, direction)
    full_sigma = float(response.norm().detach())
    direction_q, direction_scale = _quantize(direction[0].detach().float().cpu())
    return {
        "prefix_tokens": state.shape[1], "position": position,
        "local_lipschitz": float(singular[0]), "full_lipschitz_estimate": full_sigma,
        "local_singular": singular[:64].tolist(),
        "local_eigen": [[float(value.real), float(value.imag)] for value in eigen[:128]],
        "local_jacobian_q": quantized.tolist(), "local_jacobian_scale": scale,
        "token_sensitivity": sensitivity.tolist(),
        "right_direction_q": direction_q.tolist(), "right_direction_scale": direction_scale,
    }, {"local_jacobian": block_cpu, "right_singular_direction": direction.detach().cpu()}, direction

def _fisher(model: nn.Module, state: torch.Tensor, position: int, samples: int = 4):
    point = state.detach().requires_grad_(True)
    logp = model.decode(point)[0, position].log_softmax(-1)
    generator = torch.Generator().manual_seed(0)
    labels = torch.multinomial(logp.detach().exp().cpu(), samples, replacement=True,
                               generator=generator).to(state.device)
    values = []
    for label in labels:
        gradient = torch.autograd.grad(logp[label], point, retain_graph=True)[0][0, position]
        values.append(gradient.square().sum())
    return float(torch.stack(values).mean().detach())

def _step_sensitivity(model: nn.Module, state: torch.Tensor, delta: torch.Tensor):
    logits, effect = torch.func.jvp(model.decode, (state,), (delta,))
    probability = logits.detach().softmax(-1)
    effect = effect.detach()
    mean = (probability * effect).sum(-1)
    fisher_length = ((probability * effect.square()).sum(-1) - mean.square()).clamp_min(0)
    return effect.float().pow(2).mean(-1).sqrt(), fisher_length

def analyze(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor,
            states: list[torch.Tensor],
            projection_path: Path):
    projection = torch.load(projection_path, map_location="cpu", weights_only=False)
    steps = len(states) - 1
    deltas = [states[index + 1] - states[index] for index in range(steps)]
    basis = projection["basis"].to(states[0].device)
    mean = projection["mean"].to(states[0].device)
    position = min(15, inputs.shape[1] - 1)
    clouds = [((state[0] - mean) @ basis[:, :3]).cpu().tolist() for state in states]
    effective_rank = projection.get("effective_rank")
    if effective_rank is None:
        eigenvalues = projection["singular"].float().square()
        effective_rank = float(eigenvalues.sum().square() /
                               eigenvalues.square().sum().clamp_min(1e-12))
    ranks = sorted({rank for rank in (2, 4, 8, 16, 32, 64, 128, 256,
                                     round(effective_rank), basis.shape[1])
                    if rank <= basis.shape[1]})
    geometry = []
    for delta in deltas:
        vectors = delta[0]
        cumulative = (vectors @ basis).square().cumsum(-1)
        geometry.append({
            "total": vectors.square().sum(-1).cpu().tolist(),
            "parallel": torch.stack([cumulative[:, rank - 1].sqrt() for rank in ranks], -1)
                             .cpu().tolist(),
        })

    jacobian, extra, sensitive = _jacobian(model, inputs, steps, states[0])
    logit_effect = []
    fisher = []
    for state, delta in zip(states[:-1], deltas):
        effect, fisher_length = _step_sensitivity(model, state, delta)
        logit_effect.append({"rms": effect.mean().item(),
                             "rms_by_token": effect[0].cpu().tolist(),
                             "hidden_fisher_length": fisher_length.mean().item(),
                             "hidden_fisher_length_by_token": fisher_length[0].cpu().tolist()})
        fisher.append(_fisher(model, state, position))

    small_inputs = inputs[:, :16]
    small_targets = targets[:, :16]
    embedded, cos, sin = model.encode(small_inputs)
    small_states = list(model.states(small_inputs, steps, initial_state=states[0][:, :16]))
    split = steps // 2
    start = small_states[split].detach()
    remaining = steps - split
    clean = _continue(model, start, embedded, cos, sin, remaining, split)
    clean_logp = model.decode(clean).log_softmax(-1)
    clean_loss = F.nll_loss(clean_logp.reshape(-1, clean_logp.shape[-1]), small_targets.reshape(-1))
    tangent = basis[:, 0].to(start.device).expand_as(start).clone()
    random = torch.randn(start.shape, generator=torch.Generator().manual_seed(1)).to(start.device)
    random -= (random @ basis) @ basis.T
    directions = {"tangent": tangent, "normal": random, "sensitive": sensitive}
    exploration = []
    for name, direction in directions.items():
        direction = direction / direction.norm().clamp_min(1e-12)
        for amplitude in (0.01, 0.1, 1.0):
            perturbation = direction * start.norm() * amplitude
            final = _continue(model, start + perturbation, embedded, cos, sin, remaining, split)
            logp = model.decode(final).log_softmax(-1)
            loss = F.nll_loss(logp.reshape(-1, logp.shape[-1]), small_targets.reshape(-1))
            exploration.append({
                "direction": name, "amplitude": amplitude,
                "loss_change": float((loss - clean_loss).detach()),
                "kl": float(token_kl(clean_logp, logp).mean().detach()),
                "recovery": float(((final - clean).norm() / perturbation.norm()).detach()),
            })

    singular = projection["singular"].float()
    energy = singular.square()
    reconstruction = torch.sqrt(torch.clamp(energy.sum() - energy.cumsum(0), min=0) /
                                energy.sum().clamp_min(1e-12))
    return {
        "projection": {"position": position, "clouds": clouds,
                       "reference": projection["reference"].tolist(), "ranks": ranks,
                       "effective_rank": effective_rank,
                       "limits": projection["limits"].tolist(),
                       "spectrum": singular.tolist(),
                       "reconstruction": reconstruction.tolist(),
                       "delta_geometry": geometry},
        "jacobian": jacobian,
        "logit_directional_effect": logit_effect,
        "sampled_hidden_fisher_trace": fisher,
        "sampled_hidden_fisher_labels": 4,
        "exploration": exploration,
    }, extra

def rebuild(run: Path, block: torch.Tensor, tok, device: str,
            projection: Path = Path("runs/projection.pt")):
    summaries = []
    for checkpoint in sorted((run / "snapshots").glob("model_step*.pt")):
        model, blob = load(checkpoint, device)
        torch.manual_seed(10_000 + blob["step"])
        summaries.append(write_snapshot(model, block, tok, run, blob["step"], device, projection))
    (run / "diag.jsonl").write_text("".join(json.dumps(row) + "\n" for row in summaries))
    return summaries

def load(checkpoint: str | Path, device: str = "cpu"):
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    model = LoopedLM(Config(**blob["model_config"])).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, blob
