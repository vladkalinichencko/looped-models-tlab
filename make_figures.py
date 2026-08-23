"""Selection curves and the token-matched cut for the final A100 comparison."""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path("runs/a100")
CUT = 24_584_192
SERIES = [
    ("baseline-a100", "Qwen3, один проход (r=1)", "#1d4ed8"),
    ("huginn-a100", "Huginn (r=16)", "#0f766e"),
    ("antisymmetric-a100", "Антисимметричный (r=16)", "#c2410c"),
]


def load(tag):
    rows = json.loads((RUNS / f"{tag}-history").read_text())
    return [r["tokens"] for r in rows], [math.exp(r["loss"]) for r in rows]


def main():
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for tag, label, color in SERIES:
        tokens, ppl = load(tag)
        axes[0].plot([t / 1e6 for t in tokens], ppl, color=color, label=label, linewidth=2)
    axes[0].axvline(CUT / 1e6, color="#64748b", linestyle="--", linewidth=1)
    axes[0].text(CUT / 1e6, min(min(load(t)[1]) for t, _, _ in SERIES) * 1.15,
                 " срез 24.6M", color="#64748b", fontsize=9)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("обработано train-токенов, млн")
    axes[0].set_ylabel("selection perplexity (log)")
    axes[0].set_title("Динамика обучения")
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    axes[0].grid(alpha=0.25)

    names, values, colors = [], [], []
    for tag, label, color in SERIES:
        tokens, ppl = load(tag)
        cut = max((t, p) for t, p in zip(tokens, ppl) if t <= CUT)
        names.append(label.split(" (")[0])
        values.append(cut[1])
        colors.append(color)
    bars = axes[1].bar(names, values, color=colors, width=0.55)
    for bar, value in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}",
                     ha="center", va="bottom", fontsize=10)
    axes[1].set_ylabel("selection perplexity")
    axes[1].set_title(f"Token-matched срез, {CUT:,} токенов".replace(",", " "))
    axes[1].tick_params(axis="x", labelrotation=12)
    axes[1].grid(alpha=0.25, axis="y")

    figure.tight_layout()
    out = Path("assets/a100-comparison.png")
    figure.savefig(out, dpi=160)
    print(out)


if __name__ == "__main__":
    main()
