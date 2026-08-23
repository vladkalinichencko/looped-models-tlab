"""Final A100 comparison: one process per GPU, one shared token cache."""

import json
import os
from pathlib import Path
import subprocess
import sys

import torch

import data
from model import Config as ModelConfig, LoopedLM
from train import Config as TrainConfig, train

TOKENS = int(os.environ.get("LOOPED_TOKENS", 50_000_000))
RECURRENCE = 16

GROUPS = {
    "0": [
        (ModelConfig(vocab_size=16_384, n_core=4, mean_recurrence=1), "baseline-a100"),
        (ModelConfig(vocab_size=16_384, method="huginn", n_prelude=1, n_core=2, n_coda=1,
                     mean_recurrence=RECURRENCE, backprop_last=4), "huginn-a100"),
    ],
    "1": [
        (ModelConfig(vocab_size=16_384, method="antisymmetric", n_core=4,
                     mean_recurrence=RECURRENCE), "antisymmetric-a100"),
        (ModelConfig(vocab_size=16_384, method="controller", n_core=4,
                     mean_recurrence=RECURRENCE), "controller-a100"),
    ],
}


def prepare_shared():
    print("[stage] tokenizer", flush=True)
    tok = data.tokenizer()
    print("[stage] token cache", TOKENS, flush=True)
    prepared, manifest = data.prepare(tok, data.Config(train_tokens=TOKENS))
    print("[stage] token cache ready", manifest, flush=True)
    return tok, prepared, manifest


def run_group(group: str):
    tok, prepared, manifest = prepare_shared()
    for model_cfg, tag in GROUPS[group]:
        out = Path("runs") / tag / "run.json"
        if out.exists() and json.loads(out.read_text()).get("status") == "completed":
            print(f"[stage] {tag} already completed", flush=True)
            continue
        print(f"[stage] train {tag}", flush=True)
        model_cfg = ModelConfig(**{**model_cfg.__dict__, "vocab_size": len(tok)})
        torch.manual_seed(0)
        result = train(LoopedLM(model_cfg), prepared, manifest, tok,
                       TrainConfig(tag=tag, tokens=TOKENS, seed=0, eval_every=500,
                                   log_every=50, device="cuda"))
        print(f"[stage] done {tag} {result}", flush=True)


def main():
    if len(sys.argv) > 1:
        run_group(sys.argv[1])
        return
    print("[stage] building shared token cache once before forking", flush=True)
    prepare_shared()
    procs = []
    for group in GROUPS:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=group)
        log = open(f"group{group}.log", "w")
        procs.append((group, subprocess.Popen(
            [sys.executable, "run_a100.py", group], env=env, stdout=log, stderr=subprocess.STDOUT)))
        print(f"[stage] launched group {group} on CUDA device {group}", flush=True)
    for group, proc in procs:
        code = proc.wait()
        print(f"[stage] group {group} exited with {code}", flush=True)
        print(Path(f"group{group}.log").read_text()[-4000:], flush=True)


if __name__ == "__main__":
    main()
