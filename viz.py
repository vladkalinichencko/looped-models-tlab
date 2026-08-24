"""Render saved held-out diagnostics as one self-contained HTML report."""

import json
import os
from pathlib import Path


STYLE = """
:root{--ink:#18212a;--muted:#65717c;--paper:#f4f3ef;--panel:#fff;--line:#d9d6cf;--blue:#1769aa;--orange:#d16438;--green:#29845d;--purple:#7756a6;--red:#b43b4d}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}header{padding:18px 24px 12px;background:#fff;border-bottom:1px solid var(--line)}h1{margin:0;font-size:22px}h2{margin:0 0 5px;font-size:19px}h3{margin:0 0 8px;font-size:15px}.controls{position:sticky;top:0;z-index:5;display:flex;gap:20px;align-items:end;padding:10px 24px;background:#fffdf9ee;backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}.controls label{display:grid;gap:3px;font-size:12px;color:var(--muted)}.controls select,.controls input{min-width:145px}.controls output{color:var(--ink);font-variant-numeric:tabular-nums}.surface{max-width:1500px;margin:auto;padding:22px 24px 50px}.section{margin-bottom:30px}.intro{margin:0 0 14px;color:var(--muted);max-width:105ch}.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}.three{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.panel{min-width:0;padding:16px;background:var(--panel);border:1px solid var(--line)}.panel>svg,.panel>canvas{display:block;width:100%}.panel canvas{height:300px;image-rendering:pixelated}.plot{height:235px}.geometry{height:480px}.architecture{height:310px}.caption{margin:8px 0 0;color:var(--muted);max-width:92ch}.finding{padding:13px 15px;border-left:4px solid var(--blue);background:#edf4f9}.numbers{font-variant-numeric:tabular-nums}.links{display:flex;gap:14px;flex-wrap:wrap;margin-top:10px}a{color:var(--blue)}pre{margin:0;white-space:pre-wrap;max-height:160px;overflow:auto;background:#f5f4f0;padding:10px}.axis{stroke:#9ca5ad}.grid{stroke:#e7e4df}.line{fill:none;stroke-width:2.2}.band{opacity:.14}.small{font-size:11px}.direct{font-size:12px;font-weight:600}.empty{display:grid;place-items:center;min-height:170px;padding:20px;color:var(--muted);background:#f5f4f0;text-align:center}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:14px 0 28px}.card{padding:12px;background:#fff;border:1px solid var(--line)}.card b{display:block;font-size:20px;font-variant-numeric:tabular-nums}.card span{color:var(--muted);font-size:12px}.screening{margin-bottom:30px}.screening-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.screening-run{padding:11px;background:#fff;border:1px solid var(--line)}.screening-run strong{display:block}.screening-run svg{display:block;width:100%;height:46px;margin:5px 0}.screening-run a{font-size:12px;margin-right:10px}@media(max-width:900px){.two,.three,.cards,.screening-grid{grid-template-columns:1fr}.controls{overflow-x:auto;padding-left:14px}.surface{padding:16px 14px 40px}.geometry{height:390px}}
"""


SCRIPT = r"""
const D=__DATA__, C={blue:'#1769aa',orange:'#d16438',green:'#29845d',purple:'#7756a6',red:'#b43b4d'};
const svgTags=new Set(['svg','g','line','path','circle','rect','polygon','text']);
const el=(tag,a={},c=[])=>{const n=document.createElementNS(svgTags.has(tag)?'http://www.w3.org/2000/svg':'http://www.w3.org/1999/xhtml',tag);for(const[k,v]of Object.entries(a))n.setAttribute(k,v);for(const x of c)n.append(x?.nodeType?x:document.createTextNode(x??''));return n};
const fmt=x=>x==null?'—':Math.abs(x)>=100?x.toFixed(1):Math.abs(x)>=1?x.toFixed(3):x.toExponential(2);
const finite=x=>Number.isFinite(x), empty=t=>el('div',{class:'empty'},[t]);
const panel=(title,node,caption='')=>el('div',{class:'panel'},[el('h3',{},[title]),node,caption?el('p',{class:'caption'},[caption]):'']);
const section=(title,intro,nodes)=>el('section',{class:'section'},[el('h2',{},[title]),el('p',{class:'intro'},[intro]),...nodes]);
function stats(values){const a=values.filter(finite).sort((x,y)=>x-y);if(!a.length)return{mean:null,lo:null,hi:null};const q=p=>a[Math.min(a.length-1,Math.floor(p*(a.length-1)))];return{mean:a.reduce((x,y)=>x+y,0)/a.length,lo:q(.25),hi:q(.75)}}
const lineData=(values,color,name)=>({name,color,points:values.map((y,x)=>({x,y}))});
const bands=(rows,key,color,name)=>({name,color,points:rows.map((r,i)=>({x:r.step??i,...stats(r[key]||[])})).map(p=>({x:p.x,y:p.mean,lo:p.lo,hi:p.hi}))});

function chart(series,{xLabel='',yLabel='',xNames=null,marker=null}={}){
  const points=series.flatMap(s=>s.points).filter(p=>finite(p.y)),out=el('svg',{class:'plot',viewBox:'0 0 680 245'});if(!points.length)return empty('Эта величина не записана.');
  let lo=Math.min(...points.flatMap(p=>[p.lo,p.y].filter(finite))),hi=Math.max(...points.flatMap(p=>[p.hi,p.y].filter(finite)));if(lo===hi){lo-=.5;hi+=.5}const maxX=Math.max(1,...points.map(p=>p.x)),X=x=>60+x/maxX*535,Y=y=>190-(y-lo)/(hi-lo)*150;
  for(let i=0;i<4;i++){const y=40+i*50;out.append(el('line',{x1:60,y1:y,x2:595,y2:y,class:'grid'}))}out.append(el('line',{x1:60,y1:190,x2:595,y2:190,class:'axis'}),el('text',{x:8,y:42,class:'small'},[fmt(hi)]),el('text',{x:8,y:193,class:'small'},[fmt(lo)]),el('text',{x:330,y:235,'text-anchor':'middle',class:'direct'},[xLabel]),el('text',{x:13,y:125,transform:'rotate(-90 13 125)','text-anchor':'middle',class:'direct'},[yLabel]));
  if(marker!=null)out.append(el('line',{x1:X(marker),y1:28,x2:X(marker),y2:190,stroke:C.orange,'stroke-dasharray':'5 4'}),el('text',{x:X(marker)-5,y:34,fill:C.orange,class:'small','text-anchor':'end'},['selected checkpoint']));
  series.forEach((s,si)=>{const p=s.points.filter(x=>finite(x.y));if(p.some(x=>finite(x.lo))){const upper=p.map((x,i)=>(i?'L':'M')+X(x.x)+' '+Y(x.hi)).join(' '),lower=p.slice().reverse().map(x=>'L'+X(x.x)+' '+Y(x.lo)).join(' ');out.append(el('path',{d:upper+lower+'Z',fill:s.color,class:'band'}))}out.append(el('path',{d:p.map((x,i)=>(i?'L':'M')+X(x.x)+' '+Y(x.y)).join(' '),stroke:s.color,class:'line'}));p.forEach(x=>out.append(el('circle',{cx:X(x.x),cy:Y(x.y),r:3,fill:s.color})));if(p.length)out.append(el('text',{x:603,y:38+si*15,fill:s.color,class:'direct'},[s.name]))});
  [...new Set([0,Math.round(maxX/2),maxX])].forEach(x=>out.append(el('text',{x:X(x),y:207,'text-anchor':'middle',class:'small'},[xNames?.[x]??String(x)])));return out;
}
function heatmap(values,signed=false){const rows=values.length,cols=values[0]?.length||0,c=el('canvas',{width:cols,height:rows});if(!rows||!cols)return c;const ctx=c.getContext('2d'),im=ctx.createImageData(cols,rows),flat=values.flat(),max=Math.max(...flat.map(Math.abs),1e-12);flat.forEach((v,i)=>{const q=v/max,rgb=signed?(q>0?[196,64,73]:[48,93,169]):[30,104,156],a=signed?Math.abs(q):Math.sqrt(Math.abs(q));im.data.set([255-(255-rgb[0])*a,255-(255-rgb[1])*a,255-(255-rgb[2])*a,255],4*i)});ctx.putImageData(im,0,0);return c}

function architecture(model){const s=el('svg',{class:'architecture',viewBox:'0 0 900 310'}),box=(x,y,w,label,color='#e7ecef')=>{s.append(el('rect',{x,y,width:w,height:65,rx:5,fill:color,stroke:'#687581'}),el('text',{x:x+w/2,y:y+38,'text-anchor':'middle',class:'direct'},[label]));return x+w},arrow=(x1,x2,y,label='')=>{s.append(el('line',{x1,y1:y,x2,y2:y,stroke:'#687581'}),el('polygon',{points:`${x2},${y} ${x2-8},${y-4} ${x2-8},${y+4}`,fill:'#687581'}));if(label)s.append(el('text',{x:(x1+x2)/2,y:y-9,'text-anchor':'middle',class:'small'},[label]))};
  if(model.method==='baseline'){let x=20;x=box(x,105,90,'embedding');const start=x+35;for(let i=0;i<4;i++){arrow(x,x+35,137);x=box(x+35,95,130,`Qwen block ${i}`,i%2?'#eaf3ee':'#e7eef5')}const end=x;arrow(x,x+35,137);box(x+35,105,85,'LM head');if(model.mean_recurrence>1)s.append(el('path',{d:`M${end} 95 C${end} 35 ${start} 35 ${start} 95`,fill:'none',stroke:C.green,'stroke-width':2}),el('text',{x:(start+end)/2,y:25,'text-anchor':'middle',fill:C.green,class:'direct'},[`весь стек повторяется ${model.mean_recurrence} раз`]));s.append(el('text',{x:450,y:230,'text-anchor':'middle',class:'direct'},[model.mean_recurrence>1?'Четыре разных блока образуют один повторяемый цикл.':'Один проход через четыре блока с разными весами.']))}
  else if(model.method==='antisymmetric'){box(20,105,100,'embedding');arrow(120,180,137);box(180,95,250,'A = W − Wᵀ; sigmoid gate','#e7eef5');arrow(430,500,127,'(1−γ)h + gate·Ah');box(500,105,100,'state');arrow(600,680,137);box(680,105,100,'LM head');s.append(el('path',{d:'M600 105 C600 35 180 35 180 95',fill:'none',stroke:C.green,'stroke-width':2}),el('text',{x:390,y:25,'text-anchor':'middle',fill:C.green,class:'direct'},[`переход повторяется ${model.mean_recurrence} раз`]))}
  else if(model.method==='controller'){box(20,45,100,'input e');arrow(120,175,77);box(175,35,150,'GRU history','#f6eadf');arrow(325,380,67);box(380,35,120,'softmax 4');box(20,190,100,'state');for(let i=0;i<4;i++){box(150+i*145,180,115,`Qwen block ${i}`,i%2?'#eaf3ee':'#e7eef5')}box(755,180,115,'weighted sum');s.append(el('text',{x:450,y:285,'text-anchor':'middle',class:'direct'},[`softmax action returns to GRU; ${model.mean_recurrence} cycles`]))}
  else{box(20,55,85,'embedding');box(20,195,85,'random s₀');arrow(105,150,87);arrow(105,150,227);box(150,95,130,'concat adapter','#f6eadf');arrow(280,315,127);box(315,95,125,'core block 0','#e7eef5');arrow(440,475,127);box(475,95,125,'core block 1','#eaf3ee');arrow(600,635,127);box(635,95,105,'RMSNorm','#eee9f5');arrow(740,795,127);box(795,95,85,'coda');s.append(el('path',{d:'M740 95 C735 20 155 20 155 95',fill:'none',stroke:C.green,'stroke-width':2}),el('polygon',{points:'155,95 150,84 160,84',fill:C.green}),el('text',{x:445,y:25,'text-anchor':'middle',fill:C.green,class:'direct'},['этот же adapter + 2 core blocks + RMSNorm повторяется']),el('text',{x:445,y:270,'text-anchor':'middle',fill:C.orange,class:'direct'},['gradient проходит через последние 4 повтора']))}return s}
function training(run,s){const rows=run.metrics,series=[{name:'train objective',color:C.blue,points:rows.filter(x=>x.type==='train').map(x=>({x:x.step,y:x.loss}))},{name:'selection CE',color:C.orange,points:rows.filter(x=>x.type==='selection').map(x=>({x:x.step,y:x.loss}))},{name:'block objective',color:C.green,points:rows.filter(x=>finite(x.block_contrastive)).map(x=>({x:x.step,y:x.block_contrastive}))},{name:'controller CE',color:C.purple,points:rows.filter(x=>finite(x.controller_teacher_forcing)).map(x=>({x:x.step,y:x.controller_teacher_forcing}))}];return chart(series,{xLabel:'optimizer step',yLabel:'loss',marker:s.step})}
function depth(run){const rows=run.depth?.rows||[];if(!rows.length)return empty('Fixed-depth evaluation не записан.');const baseline=run.config.model.method==='baseline';return chart([{name:'selection CE',color:C.blue,points:rows.map((r,i)=>({x:i,y:r.loss}))}],{xLabel:baseline?'число повторов всего 4-block stack':'число применений recurrent core',yLabel:'cross-entropy loss',xNames:rows.map(x=>x.recurrence)})}
function outcome(run){const a=run.depth?.rows;if(!a)return 'Fixed-depth evaluation не записан.';return `${run.config.model.method}, train depth ${run.config.model.mean_recurrence}. Eval: ${a.map(x=>`${x.recurrence}:${x.loss.toFixed(3)}`).join(' · ')}.`}
function internalOutcome(run,s){const first=s.rows[0],second=s.rows[Math.min(1,s.rows.length-1)],last=s.rows.at(-1),gradient=s.rows.findIndex(r=>r.state_grad_by_token);return `${s.rows.length} cycles. Update norm: ${fmt(first.delta_norm)} → ${fmt(second.delta_norm)} → ${fmt(last.delta_norm)}. Ненулевой gradient начинается с repeat ${gradient+1}.`}
function cards(s){const last=s.rows.at(-1),eff=s.projection?.effective_rank;return el('div',{class:'cards'},[['mean token loss',last?.token_loss],['mean adjacent KL',last?.token_kl],['mean update norm',last?.delta_norm],['fixed baseline reference rank',eff]].map(([name,v])=>el('div',{class:'card'},[el('b',{},[fmt(v)]),el('span',{},[name])])))}

function loopCharts(s){const rows=s.rows;return el('div',{class:'three'},[
  panel('Token loss',chart([bands(rows,'loss_by_token',C.orange,'mean')],{xLabel:'recurrent state',yLabel:'cross-entropy'}),'Линия означает среднее по 128 held-out токенам, полоса означает межквартильный интервал.'),
  panel('Entropy',chart([bands(rows,'entropy_by_token',C.purple,'mean')],{xLabel:'recurrent state',yLabel:'nats'}),'Ось x показывает применение recurrent core. Для baseline есть один переход через весь стек.'),
  panel('KL между соседними states',chart([bands(rows,'kl_by_token',C.red,'mean')],{xLabel:'recurrent state',yLabel:'KL, nats'}),'KL показывает, насколько очередной repeat меняет распределение следующего токена.'),
  panel('Норма recurrent state',chart([bands(rows,'state_norm_by_token',C.blue,'mean')],{xLabel:'recurrent state',yLabel:'L2 norm'}),'Это норма hidden-вектора после очередного repeat, усреднённая по токенам.'),
  panel('Норма приращения state',chart([bands(rows,'delta_norm_by_token',C.green,'mean')],{xLabel:'recurrent state',yLabel:'L2 norm'}),'Это длина вектора sₜ − sₜ₋₁ в полном 384-мерном пространстве.'),
  panel('Радиальная и тангенциальная части',chart([bands(rows,'radial_update_by_token',C.red,'radial'),bands(rows,'tangent_update_by_token',C.blue,'tangent')],{xLabel:'recurrent state',yLabel:'L2 norm'}),'Радиальная часть меняет длину state. Тангенциальная часть меняет направление.'),
  panel('Cosine соседних states',chart([bands(rows,'state_cosine_by_token',C.blue,'mean')],{xLabel:'recurrent state',yLabel:'cos(sₜ₋₁, sₜ)'}),'Одинаковая длина не означает одинаковый вектор. Cosine показывает поворот state при почти фиксированной норме.'),
  panel('Градиент по входу repeat',chart([bands(rows,'state_grad_by_token',C.orange,'mean')],{xLabel:'forward repeat, gradient идёт справа налево',yLabel:'gradient L2 norm'}),'Loss находится справа после последнего repeat. Поэтому gradient обычно больше ближе к нему. Пустые первые repeats отсечены truncated backprop.')]);}
function recurrenceMechanics(s){const rows=s.rows;if(!rows[0]?.adapter_input_by_token)return empty('У baseline нет concat adapter и повторной инъекции входа.');const state=bands(rows,'adapter_state_by_token',C.blue,'из прошлого state'),input=bands(rows,'adapter_input_by_token',C.orange,'из фиксированного e'),first=rows[0],last=rows.at(-1);return el('div',{class:'two'},[panel('Что реально входит в adapter',chart([state,input],{xLabel:'repeat',yLabel:'L2 norm до сложения'}),'Adapter вычисляет Wₛsₜ + Wₑe. Обе части присутствуют. Их нормы показаны отдельно, поэтому видно, какая часть доминирует.'),panel('State norm и update norm измеряют разное',el('div',{class:'finding'},[`После RMSNorm длина state почти фиксирована: ${fmt(first.state_norm)} → ${fmt(last.state_norm)}. Расстояние между соседними точками уменьшается: ${fmt(first.delta_norm)} → ${fmt(last.delta_norm)}. Это движение по сфере, которое сходится к одной точке.`]),'Для двух векторов одинаковой длины R расстояние равно √(2R²(1 − cosine)). Норма отвечает за радиус, update norm отвечает за перемещение по сфере.')])}
function layerCharts(s){const rows=s.layers,names=rows.map(x=>`r${x.loop}:${x.name.replace('core block ','b')}`),opts={xLabel:'physical operation, r = repeat',xNames:names};return el('div',{class:'three'},[
  panel('Update внутри блоков',chart([bands(rows,'delta_norm_by_token',C.green,'mean')],{...opts,yLabel:'L2 norm'}),'Среднее и межквартильный интервал по токенам после каждой реально выполненной операции.'),
  panel('Probe loss внутри блоков',chart([bands(rows,'loss_by_token',C.orange,'mean')],{...opts,yLabel:'cross-entropy'}),'Это диагностический decode промежуточного тензора, а не training output. Adapter и core работают во внутреннем масштабе. Сравнивать между repeats надо точки после RMSNorm.'),
  panel('Градиент внутри блоков',chart([bands(rows,'state_grad_by_token',C.purple,'mean')],{...opts,yLabel:'gradient L2 norm'}),'Для Huginn повторяются одни и те же блоки. Для baseline b0…b3 имеют разные веса.')]);}
function phaseCharts(s){const colors=[C.blue,C.orange,C.green,C.purple,C.red],names=[...new Set(s.layers.map(x=>x.name))],series=key=>names.map((name,i)=>bands(s.layers.filter(x=>x.name===name).map(x=>({...x,step:x.loop})),key,colors[i%colors.length],name.replace('core block ','block ')));return el('div',{class:'three'},[
  panel('Absolute update того же слоя',chart(series('phase_delta_norm_by_token'),{xLabel:'цикл',yLabel:'‖hₜ−hₜ₋₁‖'}),'Падение к нулю означает сходимость в исходном hidden-пространстве. Рост означает расхождение.'),
  panel('Relative update того же слоя',chart(series('phase_relative_by_token'),{xLabel:'цикл',yLabel:'‖hₜ−hₜ₋₁‖ / ‖hₜ₋₁‖'}),'Падение к нулю означает сходимость этой фазы. Рост означает, что одинаковый слой уходит дальше от состояния прошлого цикла.'),
  panel('State norm по слоям',chart(series('state_norm_by_token'),{xLabel:'цикл',yLabel:'L2 norm'}),'Норма показывает радиальный рост. Она не заменяет relative update.'),
  panel('Probe loss по слоям',chart(series('loss_by_token'),{xLabel:'цикл',yLabel:'cross-entropy'}),'Loss декодирует каждую фазу одним и тем же coda и head, поэтому видно, сопровождается ли движение улучшением prediction.')]);}

function geometry(s,step,zoom=false){const p=s.projection,now=p.clouds[step],before=p.clouds[Math.max(0,step-1)],ref=p.reference,points=before.concat(now),raw=zoom?[[Math.min(...points.map(q=>q[0])),Math.min(...points.map(q=>q[1]))],[Math.max(...points.map(q=>q[0])),Math.max(...points.map(q=>q[1]))]]:p.limits,span=[raw[1][0]-raw[0][0]||1,raw[1][1]-raw[0][1]||1],lim=[[raw[0][0]-.08*span[0],raw[0][1]-.08*span[1]],[raw[1][0]+.08*span[0],raw[1][1]+.08*span[1]]],out=el('svg',{class:'geometry',viewBox:'0 0 820 480'}),dx=lim[1][0]-lim[0][0],dy=lim[1][1]-lim[0][1],X=x=>70+(x-lim[0][0])/dx*670,Y=y=>410-(y-lim[0][1])/dy*340,mean=a=>a.reduce((m,q)=>[m[0]+q[0]/a.length,m[1]+q[1]/a.length],[0,0]);for(let i=0;i<=5;i++){out.append(el('line',{x1:70+i*134,y1:70,x2:70+i*134,y2:410,class:'grid'}),el('line',{x1:70,y1:70+i*68,x2:740,y2:70+i*68,class:'grid'}))}if(!zoom)ref.forEach(q=>out.append(el('circle',{cx:X(q[0]),cy:Y(q[1]),r:1.6,fill:'#99a1a8',opacity:.25})));if(step>0)before.forEach((q,i)=>out.append(el('line',{x1:X(q[0]),y1:Y(q[1]),x2:X(now[i][0]),y2:Y(now[i][1]),stroke:C.orange,opacity:.3})));now.forEach(q=>out.append(el('circle',{cx:X(q[0]),cy:Y(q[1]),r:3,fill:C.blue,opacity:.65})));const a=mean(before),b=mean(now);out.append(el('line',{x1:X(a[0]),y1:Y(a[1]),x2:X(b[0]),y2:Y(b[1]),stroke:C.orange,'stroke-width':4}),el('circle',{cx:X(b[0]),cy:Y(b[1]),r:7,fill:C.orange}),el('text',{x:745,y:430,class:'direct'},['PC1']),el('text',{x:38,y:62,class:'direct'},['PC2']),el('text',{x:82,y:92,class:'direct',fill:C.blue},['blue: 128 token states']),el('text',{x:82,y:112,class:'direct',fill:C.orange},['orange: token updates; thick = mean']),el('text',{x:82,y:132,class:'small'},[`PC1 ${fmt(lim[0][0])} … ${fmt(lim[1][0])}; PC2 ${fmt(lim[0][1])} … ${fmt(lim[1][1])}`]));return out}
function decomposition(s,step){if(step===0)return empty('State 0 ещё не имеет update.');const p=s.projection,rank=Math.max(1,Math.round(p.effective_rank)),ri=p.ranks.reduce((best,x,i)=>Math.abs(x-rank)<Math.abs(p.ranks[best]-rank)?i:best,0),g=p.delta_geometry[step-1],inside=[],outside=[];g.total.forEach((sq,i)=>{const parallel=g.parallel[i][ri];inside.push(parallel);outside.push(Math.sqrt(Math.max(0,sq-parallel*parallel)))});const a=stats(inside),b=stats(outside),out=el('svg',{class:'plot',viewBox:'0 0 680 245'}),max=Math.max(a.hi,b.hi,1e-9),Y=x=>190-x/max*145;[['в effective subspace',a,C.blue],['вне subspace',b,C.orange]].forEach(([name,v,color],i)=>{const x=180+i*260;out.append(el('rect',{x,y:Y(v.mean),width:90,height:190-Y(v.mean),fill:color,opacity:.8}),el('line',{x1:x-12,y1:Y(v.lo),x2:x+102,y2:Y(v.lo),stroke:color,'stroke-width':2}),el('line',{x1:x-12,y1:Y(v.hi),x2:x+102,y2:Y(v.hi),stroke:color,'stroke-width':2}),el('text',{x:x+45,y:215,'text-anchor':'middle',class:'direct'},[name]),el('text',{x:x+45,y:Y(v.mean)-8,'text-anchor':'middle',class:'direct'},[fmt(v.mean)]))});out.append(el('text',{x:20,y:25,class:'direct'},[`participation-ratio rank = ${p.effective_rank.toFixed(2)}, используется rank ${p.ranks[ri]}`]));return out}

function jacobian(s){const j=s.jacobian;if(!j)return empty('Jacobian не записан.');const matrix=j.local_jacobian_q.map(r=>r.map(x=>x/127*j.local_jacobian_scale)),spectrum=lineData(j.local_singular,C.purple,'σ');return el('div',{class:'three'},[panel('Local Jacobian 384 × 384',heatmap(matrix,true),'Jᵢⱼ = ∂updateᵢ/∂stateⱼ для одного токена. Цвет показывает знак и модуль производной.'),panel('Token-to-token sensitivity 16 × 16',heatmap(j.token_sensitivity),'Колонка задаёт возмущённый токен, строка показывает токен, чей следующий state изменился.'),panel('Singular spectrum',chart([spectrum],{xLabel:'номер singular value',yLabel:'σ'}),`Максимальная локальная σ = ${fmt(j.local_lipschitz)}, оценка полного направления = ${fmt(j.full_lipschitz_estimate)}. Это локальная производная одного state.`)])}
function causal(s){const a=s.ablations||[],e=s.exploration||[];const left=a.length?chart(['freeze','reset','zero','skip'].map((name,i)=>({name,color:[C.blue,C.orange,C.red,C.purple][i],points:a.filter(x=>x.intervention===name).map(x=>({x:x.step,y:x.loss}))})),{xLabel:'repeat вмешательства',yLabel:'final token loss'}):empty('У baseline нет границы между repeats для reset, zero, freeze или skip.');const right=e.length?chart(['tangent','normal','sensitive'].map((name,i)=>({name,color:[C.blue,C.orange,C.purple][i],points:e.filter(x=>x.direction===name).map((x,j)=>({x:j,y:x.loss_change}))})),{xLabel:'амплитуда 0.01, 0.1, 1.0',yLabel:'изменение loss'}):empty('Возмущения не записаны.');return el('div',{class:'two'},[panel('Вмешательство между repeats',left,'После вмешательства модель выполняет оставшиеся repeats. Поэтому изменение final loss имеет причинный смысл.'),panel('Направление возмущения state',right,'Одинаковая относительная амплитуда сравнивает PCA-направление, ортогональное направление и чувствительное Jacobian-направление.')])}
function activeLearning(s){const a=s.logit_directional_effect||[];if(!a.length)return empty('Directional uncertainty не записана.');const improvement=s.rows.map((r,i)=>({x:r.step,y:i+1<s.rows.length?r.token_loss-s.rows[i+1].token_loss:null}));return el('div',{class:'three'},[
 panel('Ответ ещё меняется',chart([{name:'JS',color:C.red,points:s.rows.map(r=>({x:r.step,y:r.token_js}))}],{xLabel:'цикл',yLabel:'JS(pₜ,pₜ₊₁)'}),'JS измеряет изменение predictive distribution.'),
 panel('Чувствительность реального шага',chart([{name:'‖JΔh‖ RMS',color:C.blue,points:a.map((r,i)=>({x:i+1,y:r.rms}))},{name:'Fisher length',color:C.purple,points:a.map((r,i)=>({x:i+1,y:r.hidden_fisher_length}))}],{xLabel:'цикл',yLabel:'чувствительность'}),'Обе величины используют фактический update.'),
 panel('Польза следующего цикла',chart([{name:'loss improvement',color:C.green,points:improvement}],{xLabel:'цикл',yLabel:'Lₜ − Lₜ₊₁'}),'Положительное значение означает улучшение prediction.'),
 panel('Предсказывает ли score пользу',chart(['entropy','js','logit_sensitivity','hidden_fisher_length'].map((name,i)=>({name,color:[C.orange,C.red,C.blue,C.purple][i],points:(s.active_learning||[]).map(r=>({x:r.step,y:r.correlations[name]}))})),{xLabel:'цикл',yLabel:'Pearson r по токенам'}),'Корреляция сопоставляет score каждого токена с его фактическим снижением loss на следующем цикле.')]);}
function controller(s){const rows=s.controller||[];if(!rows.length)return empty('Controller diagnostics не записаны.');const colors=[C.blue,C.orange,C.green,C.purple],series=key=>[0,1,2,3].map(k=>({name:`block ${k}`,color:colors[k],points:rows.map(r=>({x:r.step,y:r[key][k]}))}));return el('div',{class:'three'},[
 panel('Softmax routing',chart(series('weights'),{xLabel:'цикл',yLabel:'mean probability'}),'Сумма четырёх вероятностей равна 1.'),
 panel('Oracle teacher action',chart(series('oracle_frequency'),{xLabel:'цикл',yLabel:'token fraction'}),'Oracle выбирает блок с минимальным token CE.'),
 panel('Controller против oracle',chart([{name:'accuracy',color:C.purple,points:rows.map(r=>({x:r.step,y:r.oracle_accuracy}))},{name:'routing entropy',color:C.orange,points:rows.map(r=>({x:r.step,y:r.entropy}))}],{xLabel:'цикл',yLabel:'value'}),'Accuracy проверяет teacher forcing; entropy показывает collapse.'),
 panel('Branch task loss',chart(series('branch_loss'),{xLabel:'цикл',yLabel:'token CE'}),'Каждый блок получает собственный task objective.'),
 panel('Cosine выходов блоков',chart([{name:'0·1',color:C.blue,points:rows.map(r=>({x:r.step,y:r.pairwise_cosine[0][1]}))},{name:'0·2',color:C.orange,points:rows.map(r=>({x:r.step,y:r.pairwise_cosine[0][2]}))},{name:'0·3',color:C.green,points:rows.map(r=>({x:r.step,y:r.pairwise_cosine[0][3]}))}],{xLabel:'цикл',yLabel:'cosine'}),'Падение cosine означает, что contrastive term разводит proposals.')]);}

const runs=Object.values(D.runs),run=document.querySelector('#run'),checkpoint=document.querySelector('#checkpoint'),loop=document.querySelector('#loop'),loopOut=document.querySelector('#loop-value'),readout=document.querySelector('#readout'),app=document.querySelector('#app');runs.forEach((r,i)=>run.append(el('option',{value:i},[r.label||r.tag])));
function refill(){const r=runs[+run.value];checkpoint.replaceChildren(...r.diag.map((s,i)=>el('option',{value:i},[`step ${s.step}`])));checkpoint.value=r.diag.length-1;configure()}
function configure(){const s=runs[+run.value].diag[+checkpoint.value];loop.max=s.rows.length;loop.value=Math.min(2,s.rows.length);draw()}
function draw(){const r=runs[+run.value],s=r.diag[+checkpoint.value],step=+loop.value,last=s.rows[Math.max(0,step-1)];loopOut.value=step;readout.value=`mean loss ${fmt(last?.token_loss)} · KL ${fmt(last?.token_kl)} · effective rank ${fmt(s.projection?.effective_rank)}`;app.replaceChildren();
  app.append(el('div',{class:'finding'},[outcome(r),el('br'),internalOutcome(r,s)]),cards(s));
  app.append(section('Что именно выполняет модель','Блок на схеме означает физическую операцию. Зелёная обратная стрелка означает повторение тех же весов, а не новый слой.',[el('div',{class:'two'},[panel('Архитектура',architecture(r.config.model)),panel('Обучение на Mac',training(r,s),'Ось x показывает optimizer step, ось y показывает cross-entropy. Синяя линия является train loss, оранжевая линия является selection loss, пунктир отмечает checkpoint диагностики.')]),el('div',{class:'links'},Object.entries(r.artifacts).map(([n,p])=>el('a',{href:p},[n])).concat([el('a',{href:s.tensor_path},['saved tensors'])]))]));
  app.append(section('Что происходит при дополнительных циклах','Fixed-depth оценка использует один checkpoint и один selection split. Для baseline повторяется весь четырёхслойный стек, для Huginn повторяется recurrent core.',[panel('Selection loss против числа циклов',depth(r),outcome(r))]));
  app.append(section('Динамика по всем токенам','Каждый график показывает всю последовательность вычисления. Линия является средним по 128 реальным held-out токенам, полоса является межквартильным интервалом. Ничего выбирать по токенам не нужно.',[loopCharts(s)]));
  app.append(section('Active Learning signals','Каждая величина рассчитана на том же реальном update и сопоставлена с фактическим изменением token loss.',[activeLearning(s)]));
  if(s.controller)app.append(section('Controller и четыре блока','Softmax-routing, oracle teacher action и специализация блоков показаны по всем циклам.',[controller(s)]));
  app.append(section('Сохраняется ли прошлый state','Concat не заменяет state входом. Adapter линейно смешивает прошлый state и фиксированный prelude output. График показывает, насколько сильно checkpoint использует каждую часть.',[recurrenceMechanics(s)]));
  app.append(section('Что делают физические блоки','Для baseline ось x идёт по четырём разным Qwen-блокам. Для Huginn подпись rN показывает repeat, внутри которого выполняются concat adapter, два общих core-блока и RMSNorm.',[layerCharts(s)]));
  app.append(section('Сходится ли каждый слой между циклами','Каждая линия сравнивает выход одного физического слоя с выходом того же слоя в предыдущем цикле. Это фактическая траектория полного стека, а не изолированное повторение слоя.',[phaseCharts(s)]));
  if(s.projection)app.append(section('Геометрия updates','Слайдер сверху выбирает только срез для геометрической картинки. Все тренды по repeats уже показаны целиком выше.',[el('div',{class:'three'},[panel('Общая шкала baseline и Huginn',geometry(s,step,false),'Фиксированная basis и фиксированный диапазон нужны для честного сравнения.'),panel('Тот же PCA, локальное увеличение',geometry(s,step,true),'Basis не меняется. Меняется только диапазон осей.'),panel('Update в полном 384D пространстве',decomposition(s,step),'Столбцы показывают среднее, засечки показывают межквартильный интервал.')]) ]));
  if(s.jacobian)app.append(section('Локальная геометрия update map','Матрицы и spectrum относятся к одному записанному prefix, token position и checkpoint.',[jacobian(s)]));
  app.append(section('Причинные проверки','Эти графики меняют recurrent state и продолжают тот же model path. Они проверяют, влияет ли измеренное направление или сохранённое состояние на final prediction.',[causal(s)]));
  app.append(section('Реальный held-out пример','Текст нужен как проверка того, что графики относятся к сохранённым данным, а не к синтетическому облаку.',[el('div',{class:'two'},[panel('Вход',el('pre',{},[s.text])),panel(`Decoded predictions после state ${step}`,el('pre',{},[s.decoded_predictions[Math.min(step,s.decoded_predictions.length-1)]]))]) ]));
}
run.onchange=refill;checkpoint.onchange=configure;loop.oninput=draw;refill();
"""



LABELS = {
    "baseline-a100": "Qwen3, один проход",
    "huginn-a100": "Huginn, 16 циклов",
    "antisymmetric-a100": "Антисимметричный переход, 16 циклов",
    "controller-a100-50M": "Контроллер над слоями, 16 циклов",
    "plain-loop-r16": "Голый луп, 16 циклов",
    "skew-no-norm-r16": "Кососимметричные блоки без нормализации",
    "skew-loop-norm-r16": "Кососимметричные блоки, одна нормализация на цикл",
    "cycle-probe-100k-r2": "Проверка глубины: 2 цикла",
    "cycle-probe-100k-r4": "Проверка глубины: 4 цикла",
    "cycle-probe-100k-r8": "Проверка глубины: 8 циклов",
    "cycle-probe-100k-r16": "Проверка глубины: 16 циклов",
}


def label_for(tag: str) -> str:
    return LABELS.get(tag, tag)


def collect(root: Path, out: Path, tags=None):
    runs = {}
    for config_path in sorted(root.glob("*/config.json")):
        if tags and config_path.parent.name not in tags:
            continue
        run = config_path.parent
        detail = run / "diag-best.json"
        snapshots = ([json.loads(detail.read_text())] if detail.exists() else
                     [json.loads(line) for line in (run / "diag.jsonl").read_text().splitlines()])
        for snapshot in snapshots:
            snapshot["tensor_path"] = os.path.relpath(snapshot["tensor_path"], out.parent)
        depth_path = run / "eval_selection.json"
        runs[run.name] = {
            "tag": run.name,
            "label": label_for(run.name),
            "config": json.loads(config_path.read_text()),
            "metrics": [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()],
            "diag": snapshots,
            "depth": json.loads(depth_path.read_text()) if depth_path.exists() else None,
            "artifacts": {name: os.path.relpath(run / filename, out.parent)
                          for name, filename in (("config", "config.json"),
                                                 ("log", "metrics.jsonl"),
                                                 ("best checkpoint", "best.pt"),
                                                 ("fixed-depth eval", "eval_selection.json"),
                                                 ("diagnostics", detail.name if detail.exists() else "diag.jsonl"))
                          if (run / filename).exists()},
        }
    return {"runs": runs}


def legacy_screening(root: Path, out: Path) -> str:
    names = {
        "idea_base": "Повтор всего Qwen3 stack",
        "idea_layer": "Повтор каждого слоя",
        "idea_layer_partial": "Незавершённый повтор каждого слоя",
        "idea_group": "Повтор групп блоков",
        "idea_prelude": "Prelude, core, coda",
        "idea_huginn": "Локальная concat-рекурсия",
        "idea_inject": "Concat без prelude и coda",
        "idea_add": "Add input injection",
        "idea_step": "Conditioning на номер цикла",
        "idea_norm": "RMSNorm между циклами",
        "idea_ouro": "Локальная entropy exit head",
        "idea_ponder": "Локальная geometric-prior exit head",
        "idea_progress": "Progress head",
        "idea_deep": "Deep supervision",
    }
    rows = []
    for tag, name in names.items():
        path = root / tag / "history.json"
        if path.exists():
            data = json.loads(path.read_text())
            rows.append((tag, name, data["best_val_loss"], data["best_val_ppl"], data["history"]))
    if not rows:
        return ""
    values = [point["val_loss"] for *_, history in rows for point in history]
    lo, hi = min(values), max(values)
    baseline = next(loss for tag, _, loss, _, _ in rows if tag == "idea_base")
    cards = []
    for tag, name, loss, ppl, history in sorted(rows, key=lambda row: row[2]):
        points = " ".join(
            f"{8 + i * 164 / max(1, len(history) - 1):.1f},{42 - (point['val_loss'] - lo) / (hi - lo) * 34:.1f}"
            for i, point in enumerate(history)
        )
        color = "#29845d" if loss < baseline else "#b43b4d"
        links = " ".join(
            f"<a href='{os.path.relpath(root / tag / file, out.parent)}'>{label}</a>"
            for file, label in (("history.json", "history"), ("diag.jsonl", "diagnostics"), ("ckpt.pt", "checkpoint"))
        )
        cards.append(
            f"<div class='screening-run'><strong>{name}</strong>"
            f"<span class='numbers'>loss {loss:.3f}, ppl {ppl:.1f}</span>"
            f"<svg viewBox='0 0 180 46' preserveAspectRatio='none'><polyline points='{points}' fill='none' stroke='{color}' stroke-width='2'/></svg>{links}</div>"
        )
    return ("<section class='screening'><h2>Старый 8M screening</h2>"
            "<p class='intro'>Все карточки используют один старый tokenizer и сравнимы только между собой. "
            "Зелёная кривая означает validation loss ниже старого baseline; эти числа не являются clean-результатом.</p>"
            f"<div class='screening-grid'>{''.join(cards)}</div></section>")


def render(out: Path = Path("runs/report.html"), tags=None, include_legacy: bool = False):
    if tags is None:
        tags = ["baseline-clean-mac", "huginn-clean-mac"]
    payload = collect(Path("runs"), out, tags)
    legacy = legacy_screening(Path("runs"), out) if include_legacy else ""
    html = ("<meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
            f"<title>Looped models: динамика и геометрия</title><style>{STYLE}</style>"
            "<header><h1>Looped models: что меняют дополнительные вычисления</h1></header>"
            "<div class='controls'><label>запуск<select id='run'></select></label>"
            "<label>checkpoint<select id='checkpoint'></select></label>"
            "<label>срез геометрии: state <output id='loop-value'></output>"
            "<input id='loop' type='range' min='0'></label>"
            f"<output id='readout'></output></div><main class='surface'>{legacy}<div id='app'></div></main><script>" +
            SCRIPT.replace("__DATA__", json.dumps(payload)) + "</script>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out


if __name__ == "__main__":
    print(render())
