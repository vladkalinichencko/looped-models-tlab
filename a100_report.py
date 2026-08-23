"""Самодостаточный HTML по финальным A100-прогонам из скачанных историй."""

import json
import math
from pathlib import Path

RUNS = Path("runs/a100")
OUT = Path("runs/a100-comparison.html")
VARIANTS = [
    ("baseline-a100", "Qwen3, один проход", 1, 4, "#1d4ed8"),
    ("huginn-a100", "Huginn", 16, 32, "#0f766e"),
    ("antisymmetric-a100", "Антисимметричный", 16, 64, "#c2410c"),
    ("controller-a100-50M", "Контроллер", 16, 64, "#7c3aed"),
]

STYLE = """
:root{color-scheme:light dark}
body{font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:32px;
     max-width:1100px;margin-inline:auto;background:#fff;color:#0f172a}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 8px}
p.note{color:#5f6b76;margin:0 0 24px}
table{border-collapse:collapse;width:100%;margin:12px 0}
th,td{border-bottom:1px solid #e2e8f0;padding:7px 10px;text-align:right;font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}
th{color:#5f6b76;font-weight:600}
svg{width:100%;height:auto;background:#fff;border:1px solid #e2e8f0;border-radius:8px}
.legend{display:flex;gap:18px;flex-wrap:wrap;margin:10px 0 0;color:#5f6b76;font-size:13px}
.legend span{display:flex;align-items:center;gap:6px}
.swatch{width:14px;height:3px;border-radius:2px}
@media(prefers-color-scheme:dark){body{background:#0b1220;color:#e2e8f0}
 svg{background:#0f172a;border-color:#1e293b}th,td{border-color:#1e293b}}
"""


def history(tag):
    path = RUNS / f"{tag}-history"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def curve(rows, box, xmax, ymin, ymax):
    points = []
    for row in rows:
        x = box[0] + row["tokens"] / xmax * box[2]
        span = math.log(ymax) - math.log(ymin)
        y = box[1] + box[3] - (math.log(math.exp(row["loss"])) - math.log(ymin)) / span * box[3]
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def chart(series):
    box = (60, 20, 980, 320)
    xmax = max((r["tokens"] for rows in series.values() for r in rows), default=1)
    values = [math.exp(r["loss"]) for rows in series.values() for r in rows]
    ymin, ymax = min(values) * 0.9, max(values) * 1.1
    parts = [f'<svg viewBox="0 0 1060 380" xmlns="http://www.w3.org/2000/svg">']
    for i in range(5):
        y = box[1] + box[3] * i / 4
        value = math.exp(math.log(ymax) - (math.log(ymax) - math.log(ymin)) * i / 4)
        parts.append(f'<line x1="{box[0]}" y1="{y}" x2="{box[0]+box[2]}" y2="{y}" '
                     f'stroke="#e2e8f0"/><text x="{box[0]-8}" y="{y+4}" text-anchor="end" '
                     f'font-size="11" fill="#5f6b76">{value:.0f}</text>')
    for i in range(6):
        x = box[0] + box[2] * i / 5
        parts.append(f'<text x="{x}" y="{box[1]+box[3]+20}" text-anchor="middle" '
                     f'font-size="11" fill="#5f6b76">{xmax*i/5/1e6:.0f}M</text>')
    for tag, label, _, _, color in VARIANTS:
        rows = series.get(tag)
        if rows:
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.2" '
                         f'points="{curve(rows, box, xmax, ymin, ymax)}"/>')
    parts.append(f'<text x="{box[0]}" y="{box[1]+box[3]+40}" font-size="12" '
                 f'fill="#5f6b76">обработано train-токенов</text>')
    parts.append(f'<text x="{box[0]}" y="14" font-size="12" fill="#5f6b76">'
                 f'selection perplexity, логарифмическая шкала</text></svg>')
    return "".join(parts)


def main():
    series = {tag: history(tag) for tag, *_ in VARIANTS}
    rows = []
    for tag, label, repeats, blocks, color in VARIANTS:
        data = series.get(tag)
        if not data:
            rows.append(f"<tr><td>{label}</td><td>считается</td><td>{repeats}</td>"
                        f"<td>{blocks}</td><td>—</td><td>—</td></tr>")
            continue
        last = max(data, key=lambda r: r["tokens"])
        rows.append(f"<tr><td>{label}</td><td>{last['tokens']:,}</td><td>{repeats}</td>"
                    f"<td>{blocks}</td><td>{last['loss']:.4f}</td>"
                    f"<td>{math.exp(last['loss']):.2f}</td></tr>".replace(",", " "))
    legend = "".join(f'<span><i class="swatch" style="background:{c}"></i>{l}</span>'
                     for _, l, _, _, c in VARIANTS if series.get(_))
    legend = "".join(f'<span><i class="swatch" style="background:{color}"></i>{label}</span>'
                     for tag, label, _, _, color in VARIANTS if series.get(tag))
    html = f"""<meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Looped models: финальные A100-прогоны</title><style>{STYLE}</style>
<h1>Финальные A100-прогоны</h1>
<p class="note">Один сид, bf16, общий tokenizer и data manifest, бюджет 50M train-токенов
на вариант. Selection считается каждые 250 шагов на первых 17 полных батчах.</p>
{chart(series)}
<div class="legend">{legend}</div>
<h2>Итог на полном бюджете</h2>
<table><thead><tr><th>модель</th><th>train tokens</th><th>повторов</th>
<th>применений блока</th><th>selection loss</th><th>selection ppl</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p class="note">Сравнение token-matched. По фактическому compute Huginn дороже baseline
примерно в восемь раз, антисимметричный вариант — в шестнадцать.</p>"""
    OUT.write_text(html)
    print(OUT)


if __name__ == "__main__":
    main()
