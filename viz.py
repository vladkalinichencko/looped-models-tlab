"""Build self-contained HTML from run JSON and saved held-out tensors."""

import json
import os
from pathlib import Path

import torch


STYLE = """
body{font:14px/1.45 -apple-system,system-ui,sans-serif;max-width:1180px;margin:auto;padding:24px;color:#18202a;background:#fafafa}
h1{font-size:22px}h2{font-size:17px;margin-top:30px}.muted{color:#657080}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
.card{background:white;border:1px solid #dfe3e8;border-radius:8px;padding:12px;overflow:auto}.flow{display:flex;align-items:center;gap:7px;flex-wrap:wrap}.box{border:1px solid #8aa4c0;border-radius:5px;padding:7px 9px;background:#edf5fc}.arrow{color:#657080}
table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:5px 8px;border-bottom:1px solid #e7eaee;text-align:right}th:first-child,td:first-child{text-align:left}select{padding:4px}svg{width:100%;height:220px}.axis{stroke:#c7cdd4}.line{fill:none;stroke-width:2}.point{r:3}a{color:#1769aa}
"""


SCRIPT = r"""
const D=__DATA__, colors=['#1769aa','#c45428','#2f855a','#7b4ab0','#b83280'];
const el=(t,a={},x=[])=>{const n=document.createElementNS(['svg','line','path','circle','text'].includes(t)?'http://www.w3.org/2000/svg':'http://www.w3.org/1999/xhtml',t);Object.entries(a).forEach(([k,v])=>n.setAttribute(k,v));x.forEach(c=>n.append(c.nodeType?c:document.createTextNode(c)));return n};
function chart(series,xkey,ykey){const points=series.flatMap(s=>s.rows.map(r=>[r[xkey],r[ykey]])).filter(p=>Number.isFinite(p[0])&&Number.isFinite(p[1]));const svg=el('svg');if(!points.length)return svg;let xs=points.map(p=>p[0]),ys=points.map(p=>p[1]),x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);if(x0===x1)x1++;if(y0===y1){y0-=.5;y1+=.5}const X=x=>40+(x-x0)/(x1-x0)*520,Y=y=>195-(y-y0)/(y1-y0)*170;svg.setAttribute('viewBox','0 0 580 220');svg.append(el('line',{x1:40,y1:195,x2:560,y2:195,class:'axis'}),el('line',{x1:40,y1:20,x2:40,y2:195,class:'axis'}));series.forEach((s,i)=>{const pts=s.rows.filter(r=>Number.isFinite(r[xkey])&&Number.isFinite(r[ykey]));const d=pts.map((r,j)=>(j?'L':'M')+X(r[xkey])+' '+Y(r[ykey])).join(' ');svg.append(el('path',{d,class:'line',stroke:colors[i%colors.length]}));pts.forEach(r=>svg.append(el('circle',{cx:X(r[xkey]),cy:Y(r[ykey]),class:'point',fill:colors[i%colors.length]})))});svg.append(el('text',{x:42,y:15},[y1.toFixed(3)]),el('text',{x:42,y:212},[x0.toFixed(1)]),el('text',{x:530,y:212},[x1.toFixed(1)]));return svg}
function card(title,node){return el('div',{class:'card'},[el('b',{},[title]),node])}
const app=document.querySelector('#app'), runs=Object.values(D.runs);
app.append(el('h2',{},['Архитектура']),...runs.map(r=>{const m=r.config.model,d=m.d_model;let nodes=m.method==='huginn'?[`Embedding × √${d}`,`Prelude ${m.n_prelude} → e`,`s₀ ~ TruncNormal`,`Concat [s,e] → ${d}`,`Qwen3 core ${m.n_core} + RMSNorm × r`,`Coda ${m.n_coda}`]:[`Embedding (${d})`,`Qwen3 blocks ${m.n_core} × 1`];nodes.push(`RMSNorm → tied head (${m.vocab_size})`);const flow=el('div',{class:'flow'},nodes.flatMap((n,i)=>i?[el('span',{class:'arrow'},['→']),el('span',{class:'box'},[n])]:[el('span',{class:'box'},[n])]));const links=el('p',{class:'muted'},Object.entries(r.artifacts).flatMap(([name,path],i)=>[...(i?[' · ']:[]),el('a',{href:path},[name])]));return card(r.tag,el('div',{},[flow,links]))}));
app.append(el('h2',{},['Обучение']),card('Selection perplexity по токенам',chart(runs.map(r=>({rows:r.history.map(x=>({tokens:x.tokens/1e6,ppl:x.ppl}))})),'tokens','ppl')));
const selector=el('select',{},runs.map((r,i)=>el('option',{value:i},[r.tag]))),snap=el('select'),panel=el('div');app.append(el('h2',{},['Held-out пример и recurrent states']),el('div',{},[selector,snap]),panel);
function draw(){const r=runs[+selector.value],s=r.diag[+snap.value];panel.replaceChildren();if(!s)return;panel.append(el('p',{class:'muted'},[`train step ${s.step}. Source tensors: `]),el('a',{href:s.tensor_path},[s.tensor_path]));panel.append(card('Вход',el('div',{},[s.text])));const grid=el('div',{class:'grid'});grid.append(card('Норма шага',chart([{rows:s.rows}],'step','delta_norm')),card('Cosine соседних шагов',chart([{rows:s.rows}],'step','delta_cosine')),card('Token-level KL соседних шагов',chart([{rows:s.rows}],'step','token_kl')),card('Loss по шагам',chart([{rows:s.rows}],'step','token_loss')),card('Градиент по recurrent state',chart([{rows:s.rows}],'step','state_grad')));if(s.trajectory)grid.append(card('Общая baseline projection',chart(s.trajectory.map((rows,i)=>({rows:rows.map((p,step)=>({step,x:p[0],y:p[1]}))})),'x','y')));panel.append(grid);panel.append(el('h3',{},['Decoded prediction по шагам']),el('ol',{},s.decoded_predictions.map(x=>el('li',{},[x]))));const ab=el('table',{},[el('tr',{},['step','intervention','loss','KL к исходному'].map(x=>el('th',{},[x]))),...s.ablations.map(x=>el('tr',{},[x.step,x.intervention,x.loss.toFixed(4),x.kl_to_original.toFixed(4)].map(v=>el('td',{},[String(v)]))))]);panel.append(el('h3',{},['Причинные абляции']),ab);const tok=el('table',{},[el('tr',{},['pos','input','target','prediction','loss','exit depth'].map(x=>el('th',{},[x]))),...s.tokens.map(x=>el('tr',{},[x.position,x.input,x.target,x.prediction,x.loss.toFixed(3),x.exit_depth].map(v=>el('td',{},[String(v)]))))]);panel.append(el('h3',{},['Все позиции примера']),tok)}
function snaps(){snap.replaceChildren(...runs[+selector.value].diag.map((s,i)=>el('option',{value:i},[`step ${s.step}`])));snap.value=Math.max(0,snap.options.length-1);draw()}selector.onchange=snaps;snap.onchange=draw;snaps();
"""


def collect(root: Path, out: Path, tags=None):
    projection_path = root / "projection.pt"
    projection = torch.load(projection_path, map_location="cpu", weights_only=False) if projection_path.exists() else None
    runs = {}
    for config_path in sorted(root.glob("*/config.json")):
        if not tags and config_path.parent.name.startswith("_"):
            continue
        if tags and config_path.parent.name not in tags:
            continue
        config = json.loads(config_path.read_text())
        history_path = config_path.parent / "history.json"
        diag_path = config_path.parent / "diag.jsonl"
        snapshots = [json.loads(line) for line in diag_path.read_text().splitlines()] if diag_path.exists() else []
        for snapshot in snapshots:
            snapshot["tensor_path"] = os.path.relpath(snapshot["tensor_path"], out.parent)
            if projection:
                raw = torch.load(config_path.parent / "snapshots" / f"step{snapshot['step']:06d}.pt",
                                 map_location="cpu", weights_only=False)
                snapshot["trajectory"] = []
                for position in range(min(8, raw["states"][0].shape[1])):
                    points = [((state[0, position].float() - projection["mean"]) @
                               projection["basis"]).tolist() for state in raw["states"]]
                    snapshot["trajectory"].append(points)
        runs[config_path.parent.name] = {
            "tag": config_path.parent.name, "config": config,
            "history": json.loads(history_path.read_text()) if history_path.exists() else [],
            "diag": snapshots,
            "artifacts": {name: os.path.relpath(config_path.parent / filename, out.parent)
                          for name, filename in (("config", "config.json"), ("log", "metrics.jsonl"),
                                                 ("best checkpoint", "best.pt"),
                                                 ("last checkpoint", "last.pt"),
                                                 ("diagnostics", "diag.jsonl"))},
        }
    return {"runs": runs, "projection_source": projection.get("source") if projection else None}


def render(out: Path = Path("runs/report.html"), tags=None):
    payload = collect(Path("runs"), out, tags)
    html = (f"<meta charset='utf-8'><title>Looped models</title><style>{STYLE}</style>"
            "<h1>Looped models: clean preliminary runs</h1>"
            f"<p class='muted'>Одна projection basis: {payload['projection_source'] or 'ещё не построена'}.</p>"
            "<div id='app'></div><script>" + SCRIPT.replace("__DATA__", json.dumps(payload)) + "</script>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out


if __name__ == "__main__":
    print(render())
