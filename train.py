"""One explicit training and selection-evaluation path for every model."""

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
import itertools
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

import data
import diag
import viz
from torch import nn

from methods.huginn import HuginnLoopedLM
from model import Config as ModelConfig

@dataclass(frozen=True)
class Config:
    tag: str
    tokens: int = 8_000_000
    seed: int = 0
    batch_size: int = 16
    seq_len: int = 512
    lr: float = 3e-3
    min_lr: float = 3e-4
    warmup: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 200
    log_every: int = 20
    device: str = "auto"

def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def lr_at(step: int, total: int, cfg: Config) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = (step - cfg.warmup) / max(total - cfg.warmup, 1)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))

@contextmanager
def fixed_rng(device: str, seed: int):
    device = device.split(":", 1)[0]
    cpu_state = torch.random.get_rng_state()
    accelerator_state = (torch.mps.get_rng_state() if device == "mps" else
                         torch.cuda.get_rng_state() if device == "cuda" else None)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if device == "mps":
            torch.mps.set_rng_state(accelerator_state)
        elif device == "cuda":
            torch.cuda.set_rng_state(accelerator_state)

@torch.no_grad()
def evaluate(model: nn.Module, blocks: torch.Tensor, cfg: Config) -> float:
    model.eval()
    batches = itertools.islice(
        data.batches(blocks, cfg.batch_size, next(model.parameters()).device), 17)
    losses = []
    for x, y in batches:
        logits = model(x, steps=model.cfg.mean_recurrence)[0]
        losses.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).item())
    model.train()
    return sum(losses) / len(losses)

def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")

def train(model: nn.Module, prepared: dict[str, torch.Tensor], manifest: Path,
          tok, cfg: Config):
    torch.manual_seed(cfg.seed)
    device = pick_device(cfg.device)
    model.to(device).train()
    total, non_embedding = model.n_params()
    if non_embedding > 10_000_000:
        raise ValueError(f"non-embedding parameter budget exceeded: {non_embedding}")
    if cfg.seq_len != model.cfg.max_seq or cfg.batch_size * cfg.seq_len > cfg.tokens:
        raise ValueError("training and model/data shapes disagree")

    out = Path("runs") / cfg.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "snapshots").mkdir(exist_ok=True)
    runtime = {
        "device": device,
        "dtype": "bfloat16" if device == "cuda" else "float32",
        "torch": torch.__version__,
        "params_total": total,
        "params_non_embedding": non_embedding,
        "data_manifest": str(manifest),
    }
    write_json(out / "config.json", {
        "model": model.config_dict(), "training": asdict(cfg), "runtime": runtime,
    })
    write_json(out / "run.json", {"status": "running", **runtime})
    (out / "diag.jsonl").write_text("")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=cfg.lr, betas=(0.9, 0.95))
    total_steps = cfg.tokens // (cfg.batch_size * cfg.seq_len)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    depth_rng = torch.Generator().manual_seed(cfg.seed)
    best = float("inf")
    history = []
    started = time.time()

    with (out / "metrics.jsonl").open("w") as log:
        for step, (x, y) in enumerate(data.batches(prepared["train"], cfg.batch_size, device)):
            if step >= total_steps:
                break
            lr = lr_at(step, total_steps, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            recurrence = model.sample_steps(depth_rng)
            with amp:
                loss = model(x, y, steps=recurrence)[1]
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % cfg.log_every == 0:
                row = {"type": "train", "step": step, "tokens": step * cfg.batch_size * cfg.seq_len,
                       "loss": loss.item(), "lr": lr, "recurrence": recurrence,
                       "grad_norm": grad_norm, "seconds": time.time() - started}
                row.update(getattr(model, "last_objective", {}))
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(f"{cfg.tag} {step:4}/{total_steps} loss={loss.item():.4f} r={recurrence}")

            if (step and step % cfg.eval_every == 0) or step == total_steps - 1:
                with fixed_rng(device, cfg.seed + 10_000):
                    selection_loss = evaluate(model, prepared["selection"], cfg)
                    diag.write_snapshot(model, prepared["selection"][0], tok, out, step, device)
                row = {"type": "selection", "step": step,
                       "tokens": (step + 1) * cfg.batch_size * cfg.seq_len,
                       "loss": selection_loss, "ppl": math.exp(selection_loss)}
                history.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
                checkpoint = {"model": model.state_dict(), "model_config": model.config_dict(),
                              "training_config": asdict(cfg), "runtime": runtime,
                              "step": step, "selection_loss": selection_loss,
                              "tokenizer": str(data.TOKENIZER_DIR)}
                torch.save(checkpoint, out / "snapshots" / f"model_step{step:06d}.pt")
                torch.save(checkpoint, out / "last.pt")
                if selection_loss < best:
                    best = selection_loss
                    torch.save(checkpoint, out / "best.pt")
                write_json(out / "history.json", history)
                viz.render(out / "report.html", [cfg.tag])

    result = {"status": "completed", "steps": total_steps,
              "tokens": total_steps * cfg.batch_size * cfg.seq_len,
              "best_selection_loss": best, "best_selection_ppl": math.exp(best), **runtime}
    write_json(out / "run.json", result)
    return result

if __name__ == "__main__":
    # Финальный вариант отчёта: Huginn с 16 повторами на бюджете задания.
    tok = data.tokenizer()
    prepared, manifest = data.prepare(tok, data.Config(train_tokens=50_000_000))
    model_cfg = ModelConfig(vocab_size=len(tok), method="huginn", n_prelude=1, n_core=2,
                            n_coda=1, mean_recurrence=16, backprop_last=4)
    torch.manual_seed(0)
    print(train(HuginnLoopedLM(model_cfg), prepared, manifest, tok,
                Config(tag="huginn", tokens=50_000_000, eval_every=250)))
