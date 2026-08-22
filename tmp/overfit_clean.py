import json

import torch

import data
from model import Config, LoopedLM
from train import fixed_rng


def run(model, x, y):
    model.to("mps").train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95))
    losses = []
    for _ in range(30):
        with fixed_rng("mps", 123):
            loss = model(x, y, steps=model.cfg.mean_recurrence)[1]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    return {"first": losses[0], "last": losses[-1]}


def main():
    tok = data.tokenizer()
    prepared, _ = data.prepare(tok, data.Config())
    batch = prepared["train"][:2, :129].to("mps")
    x, y = batch[:, :-1], batch[:, 1:]
    configs = {
        "baseline": Config(vocab_size=len(tok), n_core=4, max_seq=512),
        "huginn": Config(vocab_size=len(tok), method="huginn", n_prelude=1, n_core=2,
                         n_coda=1, mean_recurrence=4, backprop_last=4, max_seq=512),
    }
    result = {}
    for name, cfg in configs.items():
        torch.manual_seed(0)
        result[name] = run(LoopedLM(cfg), x, y)
        torch.mps.empty_cache()
    with open("tmp/overfit_clean.json", "w") as handle:
        json.dump(result, handle, indent=2)
    print(result)


if __name__ == "__main__":
    main()
