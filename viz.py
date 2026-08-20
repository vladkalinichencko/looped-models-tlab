"""runs/*/{history.json,diag.jsonl} -> one self-contained interactive page.

Static pictures answer one question each and then go stale. The diagnostics here are
a cube — metric x loop index x training step x run — so the page keeps the cube and
lets you slice it: pick a run, drag the training step, watch the per-loop curves move.

    python viz.py                      # -> runs/report.html
    python viz.py --out /tmp/x.html
"""

import argparse
import json
import pathlib

TEMPLATE = """<title>Looped models — диагностика</title>
<style>
:root { --bg:#fff; --fg:#111; --mut:#666; --line:#ddd; --acc:#2b6cb0; }
:root:not([data-theme=light]) { }
@media (prefers-color-scheme: dark) { :root:not([data-theme=light]) {
  --bg:#14161a; --fg:#e8e8e8; --mut:#9aa0a6; --line:#2c3038; --acc:#7aa7dd; } }
:root[data-theme=dark] { --bg:#14161a; --fg:#e8e8e8; --mut:#9aa0a6; --line:#2c3038; --acc:#7aa7dd; }
body { background:var(--bg); color:var(--fg); font:14px/1.5 -apple-system,system-ui,sans-serif;
       margin:0 auto; max-width:1180px; padding:24px; }
h1 { font-size:20px; margin:0 0 4px; } h2 { font-size:15px; margin:28px 0 8px; font-weight:600; }
p.note { color:var(--mut); margin:2px 0 14px; }
table { border-collapse:collapse; font-size:13px; } td,th { padding:3px 10px 3px 0; text-align:right; }
th:first-child,td:first-child { text-align:left; }
th { border-bottom:1px solid var(--line); font-weight:600; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(270px,1fr)); gap:14px; }
.card { border:1px solid var(--line); border-radius:6px; padding:8px 10px; overflow-x:auto; }
.card b { font-size:12px; font-weight:600; } .card span { color:var(--mut); font-size:11px; }
.ctl { display:flex; gap:14px; align-items:center; flex-wrap:wrap; margin:8px 0 4px; }
select,input[type=range] { accent-color:var(--acc); }
.legend { display:flex; gap:12px; flex-wrap:wrap; font-size:12px; color:var(--mut); margin:4px 0; }
.legend i { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; }
svg { display:block; } .ax { stroke:var(--line); } .tick { fill:var(--mut); font-size:10px; }
</style>
<h1>Looped models — что происходит с состоянием между лупами</h1>
<p class="note">Данные — runs/*/history.json и runs/*/diag.jsonl. Ничего не досчитывается на странице.</p>
<div id="app"></div>
<script>
const DATA = __DATA__;
const PAL = ["#2b6cb0","#c05621","#2f855a","#805ad5","#b83280","#4a5568","#b7791f","#2c7a7b"];

const SVG = new Set(["svg", "g", "path", "line", "text", "rect", "circle"]);
function el(tag, attrs, kids) {
  const n = document.createElementNS(SVG.has(tag)
      ? "http://www.w3.org/2000/svg" : "http://www.w3.org/1999/xhtml", tag);
  for (const k in (attrs || {})) n.setAttribute(k, attrs[k]);
  for (const c of (kids || [])) n.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  return n;
}
const fmt = v => Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0)
  ? v.toExponential(1) : (+v.toFixed(3)).toString();

function chart(series, o) {
  o = Object.assign({w: 260, h: 150, log: false, xlab: "", pad: 34}, o);
  const pts = series.flatMap(s => s.pts).filter(p => isFinite(p[1]));
  if (!pts.length) return el("svg", {width: o.w, height: o.h});
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const tr = v => o.log ? Math.log10(Math.max(v, 1e-12)) : v;
  let [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  let [y0, y1] = [Math.min(...ys.map(tr)), Math.max(...ys.map(tr))];
  if (x1 === x0) x1 = x0 + 1;
  if (y1 === y0) { y0 -= .5; y1 += .5; }
  const m = (y1 - y0) * 0.08; y0 -= m; y1 += m;
  const X = v => o.pad + (v - x0) / (x1 - x0) * (o.w - o.pad - 8);
  const Y = v => o.h - 20 - (tr(v) - y0) / (y1 - y0) * (o.h - 30);
  const g = el("svg", {width: o.w, height: o.h});
  g.appendChild(el("line", {x1: o.pad, y1: o.h - 20, x2: o.w - 8, y2: o.h - 20, class: "ax"}));
  g.appendChild(el("line", {x1: o.pad, y1: 10, x2: o.pad, y2: o.h - 20, class: "ax"}));
  for (const [v, y] of [[y1 - m, 14], [y0 + m, o.h - 22]])
    g.appendChild(el("text", {x: 2, y: y, class: "tick"}, [fmt(o.log ? Math.pow(10, v) : v)]));
  for (const [v, a] of [[x0, "start"], [x1, "end"]])
    g.appendChild(el("text", {x: X(v), y: o.h - 6, class: "tick", "text-anchor": a}, [fmt(v)]));
  series.forEach((s, i) => {
    const d = s.pts.filter(p => isFinite(p[1])).map((p, j) => (j ? "L" : "M") + X(p[0]) + " " + Y(p[1])).join(" ");
    g.appendChild(el("path", {d: d, fill: "none", "stroke-width": s.w || 1.6,
      stroke: s.color || PAL[i % PAL.length], opacity: s.o == null ? 1 : s.o}));
  });
  return g;
}

function card(title, sub, svg) {
  return el("div", {class: "card"}, [el("b", {}, [title]), sub ? el("span", {}, [" — " + sub]) : "", svg]);
}
function legend(names, colors) {
  return el("div", {class: "legend"}, names.map((n, i) =>
    el("span", {}, [el("i", {style: "background:" + (colors ? colors[i] : PAL[i % PAL.length])}), n])));
}

const app = document.getElementById("app");
const runs = Object.keys(DATA.runs);

// --- 1. параметры прогонов
{
  const rows = runs.map(r => DATA.runs[r].config);
  const cols = ["tag", "n_loops", "loop_scheme", "input_injection", "step_cond", "tokens"];
  const t = el("table", {}, [el("tr", {}, cols.concat(["non-emb", "val ppl"]).map(c => el("th", {}, [c])))]);
  runs.forEach(r => {
    const c = DATA.runs[r], p = c.best_val_ppl;
    t.appendChild(el("tr", {}, cols.map(k => el("td", {}, [String(c.config[k])]))
      .concat([el("td", {}, [fmt(c.params.non_embedding / 1e6) + "M"]),
               el("td", {}, [p == null ? "—" : fmt(p)])])));
  });
  app.appendChild(el("h2", {}, ["Прогоны"]));
  app.appendChild(t);
}

// --- 2. кривые обучения
{
  app.appendChild(el("h2", {}, ["Обучение при равном бюджете токенов"]));
  app.appendChild(el("p", {class: "note"}, ["Ось y логарифмическая. Один и тот же токенизатор, "
    + "одни и те же токены, различается только схема лупинга."]));
  const s = runs.map(r => ({pts: DATA.runs[r].history.map(h => [h.tokens / 1e6, h.val_ppl])}));
  app.appendChild(legend(runs));
  app.appendChild(card("val perplexity", "млн токенов", chart(s, {w: 560, h: 260, log: true})));
}

// --- 3. метрики по лупам, с ползунком по обучению
{
  app.appendChild(el("h2", {}, ["Состояние по шагам лупа"]));
  app.appendChild(el("p", {class: "note"}, ["Цвет — момент обучения: светлое рано, тёмное поздно. "
    + "Ползунок подсвечивает один снимок."]));
  const sel = el("select", {}, runs.filter(r => DATA.runs[r].diag.length).map(r => el("option", {}, [r])));
  const slider = el("input", {type: "range", min: 0, max: 0, value: 0, style: "width:260px"});
  const label = el("span", {class: "note"}, [""]);
  app.appendChild(el("div", {class: "ctl"}, [sel, slider, label]));
  const box = el("div", {class: "grid"});
  app.appendChild(box);

  const KEYS = [["rel_step", "‖Δh‖/‖h‖ — сходится ли к неподвижной точке"],
                ["cos_prev", "cos(Δh_t, Δh_{t-1}) — новое движение или то же самое"],
                ["cos_useful", "cos(Δh, −∂L/∂h) — полезен ли шаг"],
                ["h_norm", "‖h‖ — растёт ли поток невязки"],
                ["eff_rank", "эффективный ранг"],
                ["intrinsic_dim", "внутренняя размерность, TwoNN"],
                ["grad_in", "‖∂L/∂h‖ на входе шага — затухание через повторы"],
                ["kl_to_final", "KL(p_t ‖ p_финал) — когда предсказание замирает"],
                ["top1_changed", "доля токенов, у которых top-1 ещё изменится"],
                ["entropy", "энтропия softmax"],
                ["top1_prob", "вероятность top-1"],
                ["loss", "лосс при выходе на этом шаге"]];

  function draw(reset) {
    const d = DATA.runs[sel.value].diag;
    slider.max = d.length - 1;
    if (reset) slider.value = d.length - 1;
    const cur = Math.min(+slider.value, d.length - 1);
    label.textContent = "шаг " + d[cur].step + " · " + fmt(d[cur].tokens / 1e6) + "M токенов";
    box.textContent = "";
    for (const [k, title] of KEYS) {
      const series = d.map((snap, i) => ({
        pts: snap.rows.map(r => [r.step, r[k]]).filter(p => p[1] != null),
        color: i === cur ? PAL[0] : "#888", w: i === cur ? 2.2 : 1,
        o: i === cur ? 1 : 0.12 + 0.5 * i / Math.max(d.length - 1, 1)}));
      box.appendChild(card(k, title, chart(series, {})));
    }
  }
  sel.onchange = () => draw(true);
  slider.oninput = () => draw(false);
  draw(true);
}

// --- 4. градиент по блокам во время обучения
{
  app.appendChild(el("h2", {}, ["Градиент по блокам за время обучения"]));
  app.appendChild(el("p", {class: "note"}, ["Один и тот же блок применяется на каждом лупе, "
    + "поэтому вопрос не «затухает ли по глубине», а «достаётся ли сигнал каждому из блоков»."]));
  const box = el("div", {class: "grid"});
  for (const r of runs) {
    const d = DATA.runs[r].diag;
    if (!d.length) continue;
    const keys = Object.keys(d[0]).filter(k => k.startsWith("grad_block"));
    const s = keys.map(k => ({pts: d.map(x => [x.tokens / 1e6, x[k]])}));
    box.appendChild(card(r, "млн токенов, лог", chart(s, {log: true})));
  }
  app.appendChild(box);
  app.appendChild(legend(["block0", "block1", "block2", "block3"]));
}
</script>
"""


def collect(root):
    runs = {}
    for path in sorted(pathlib.Path(root).glob("*/history.json")):
        blob = json.loads(path.read_text())
        diag = []
        jsonl = path.parent / "diag.jsonl"
        if jsonl.exists():
            diag = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
        runs[path.parent.name] = {**blob, "diag": diag}
    return {"runs": runs}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="runs/report.html")
    p.add_argument("--only", nargs="+", default=None, help="какие прогоны показывать")
    args = p.parse_args()

    data = collect(args.runs)
    if args.only:
        data["runs"] = {k: v for k, v in data["runs"].items() if k in args.only}
    out = pathlib.Path(args.out)
    out.write_text(TEMPLATE.replace("__DATA__", json.dumps(data)))
    print(f"{len(data['runs'])} прогонов -> {out}")


if __name__ == "__main__":
    main()
