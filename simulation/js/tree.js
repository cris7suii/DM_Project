/* ---------- Arbre ---------- */
let steps=[],ns=1,maxD=3;
function build(){steps=[];const q=[{s:pts,b:[0,10,0,10],dep:0}];
 while(q.length){const n=q.shift();if(n.dep>=maxD||gini(n.s)==0||n.s.length<4)continue;let best=null;
  for(const f of['x','y'])for(let t=.5;t<10;t+=.5){const L=n.s.filter(p=>p[f]<=t),R=n.s.filter(p=>p[f]>t);if(!L.length||!R.length)continue;
   const gw=(L.length*gini(L)+R.length*gini(R))/n.s.length;if(!best||gw<best.gw)best={f,t,gw,L,R}}
  if(!best)continue;const[a,b,c,d]=n.b;steps.push({f:best.f,t:best.t,b:n.b,g0:gini(n.s),gw:best.gw,n:n.s.length});
  const bl=best.f=='x'?[a,best.t,c,d]:[a,b,c,best.t],br=best.f=='x'?[best.t,b,c,d]:[a,b,best.t,d];
  q.push({s:best.L,b:bl,dep:n.dep+1},{s:best.R,b:br,dep:n.dep+1})}
 ns=Math.min(ns,steps.length)}
function predT(p){let b=[0,10,0,10];for(let i=0;i<ns;i++){const s=steps[i];if(p.x<b[0]||p.x>b[1]||p.y<b[2]||p.y>b[3])continue;
  s.f=='x'?(p.x<=s.t?b=[b[0],s.t,b[2],b[3]]:b=[s.t,b[1],b[2],b[3]]):(p.y<=s.t?b=[b[0],b[1],b[2],s.t]:b=[b[0],b[1],s.t,b[3]])}return b}
function drawTree(){build();base();
 const regs=new Map();pts.forEach(p=>{const k=predT(p).join();(regs.get(k)||regs.set(k,{b:predT(p),s:[]}).get(k)).s.push(p)});
 let ok=0;regs.forEach(r=>{const m=r.s.filter(p=>p.c).length>=r.s.length/2?1:0;ok+=r.s.filter(p=>p.c==m).length;
  g.fillStyle=m?'rgba(16,185,129,.13)':'rgba(249,115,22,.13)';g.fillRect(sx(r.b[0]),sy(r.b[3]),sx(r.b[1])-sx(r.b[0]),sy(r.b[2])-sy(r.b[3]))});
 g.strokeStyle='#818cf8';g.lineWidth=2;steps.slice(0,ns).forEach(s=>{const[a,b,c,d]=s.b;g.beginPath();
  s.f=='x'?(g.moveTo(sx(s.t),sy(c)),g.lineTo(sx(s.t),sy(d))):(g.moveTo(sx(a),sy(s.t)),g.lineTo(sx(b),sy(s.t)));g.stroke()});g.lineWidth=1;dots();
 const s=steps[ns-1];
 $('#panel').innerHTML=`<h2>Construction pas à pas (critère de Gini)</h2>
 <div class="row">Profondeur max <input type=range min=1 max=5 value=${maxD} oninput="maxD=+this.value;drawTree()"> <b>${maxD}</b></div>
 <div class="row"><button class="btn" onclick="ns=Math.max(1,ns-1);drawTree()">◀</button>
 <button class="btn" onclick="ns=Math.min(steps.length,ns+1);drawTree()">Coupe suivante ▶</button> Étape <b>${ns}/${steps.length}</b></div>
 ${s?`<div class="note">Coupe n°${ns} : <code>${s.f=='x'?'X (technologies)':'Y (expérience)'} ≤ ${s.t}</code><br>
 Gini du nœud = ${s.g0.toFixed(3)} → Gini pondéré après coupe = <b>${s.gw.toFixed(3)}</b> (${s.n} candidats)<br>
 Gain = ${(s.g0-s.gw).toFixed(3)}</div>`:''}
 <b>Exactitude d'apprentissage : ${(ok/pts.length*100).toFixed(1)} %</b>
 <div class="note">Données « règle exacte » : la 1ʳᵉ coupe retrouve <code>X ≤ 2.5</code> (Gini→0), c'est pourquoi l'arbre atteint 99,9 % dans le projet. En profil seul, l'arbre sur-apprend le bruit.</div>`}
T.push(['🌳 Arbre de décision',drawTree,'Partition orthogonale',0]);
