import torch

from model import Config, LoopedLM


def tiny(method):
    common = dict(vocab_size=97, d_model=32, n_heads=4, n_kv_heads=2, head_dim=8,
                  d_ff=64, max_seq=16)
    if method == "baseline":
        return LoopedLM(Config(**common, n_core=2))
    return LoopedLM(Config(**common, method="huginn", n_prelude=1, n_core=2, n_coda=1,
                           mean_recurrence=4, backprop_last=2))


def main():
    x = torch.randint(0, 97, (2, 16))
    y = torch.randint(0, 97, (2, 16))
    for method in ("baseline", "huginn"):
        model = tiny(method)
        logits, loss = model(x, y, steps=6 if method == "huginn" else 1)
        loss.backward()
        assert logits.shape == (2, 16, 97)
        assert all(parameter.grad is not None for parameter in model.core.parameters())

    model = tiny("huginn")
    states = list(model.states(x, steps=6))
    for state in states:
        if state.requires_grad:
            state.retain_grad()
    model.decode(states[-1]).sum().backward()
    assert [state.grad is not None for state in states] == [False, False, False, False, True, True, True]


if __name__ == "__main__":
    main()
