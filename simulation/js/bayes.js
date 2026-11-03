/* ---------- Naive Bayes gaussien ---------- */
function nbFit(){const m=[[],[]];pts.forEach(p=>m[p.c].push(p));
 return m.map(a=>{const n=a.length||1,mx=a.reduce((s,p)=>s+p.x,0)/n,my=a.reduce((s,p)=>s+p.y,0)/n;
  return{n:a.length,mx,my,vx:a.reduce((s,p)=>s+(p.x-mx)**2,0)/n+.01,vy:a.reduce((s,p)=>s+(p.y-my)**2,0)/n+.01}})}
const lg=(v,m,s)=>-.5*Math.log(2*Math.PI*s)-(v-m)**2/(2*s);
function nbPost(M,x,y){const t=M[0].n+M[1].n,l=M.map(c=>Math.log((c.n+1e-9)/t)+lg(x,c.mx,c.vx)+lg(y,c.my,c.vy)),
 m=Math.max(...l),e=l.map(v=>Math.exp(v-m));return{p1:e[1]/(e[0]+e[1]),l}}
function drawNB(){base();const M=nbFit(),N=28,s=(W-2*P)/N;
 for(let i=0;i<N;i++)for(let j=0;j<N;j++){const p=nbPost(M,(i+.5)/N*10,(j+.5)/N*10).p1;
  g.fillStyle=`rgba(${p>.5?'16,185,129':'249,115,22'},${Math.abs(p-.5)*.5})`;g.fillRect(sx(i/N*10),sy((j+1)/N*10),s+1,s+1)}
 dots();star();const r=nbPost(M,Q.x,Q.y),t=M[0].n+M[1].n,pe=r.p1*100,f=v=>v.toFixed(2);
 $('#panel').innerHTML=`<h2>Naive Bayes gaussien — P(c|x) ∝ P(c)·P(X|c)·P(Y|c)</h2>
 <table><tr><th>Classe</th><th>P(c)</th><th>μ_X</th><th>σ²_X</th><th>μ_Y</th><th>σ²_Y</th></tr>
 <tr class="R"><td>Non employé</td><td>${f(M[0].n/t)}</td><td>${f(M[0].mx)}</td><td>${f(M[0].vx)}</td><td>${f(M[0].my)}</td><td>${f(M[0].vy)}</td></tr>
 <tr class="B"><td>Employé</td><td>${f(M[1].n/t)}</td><td>${f(M[1].mx)}</td><td>${f(M[1].vx)}</td><td>${f(M[1].my)}</td><td>${f(M[1].vy)}</td></tr></table>
 <div class="note">Candidat ★ = (${f(Q.x)} ; ${f(Q.y)}) — log-vraisemblances : non employé = ${f(r.l[0])} · employé = ${f(r.l[1])}</div>
 <div class="bar"><i style="width:${pe}%;background:var(--b)"></i><i style="width:${100-pe}%;background:var(--r)"></i></div>
 <b>P(employé | x) = ${pe.toFixed(1)} %</b> → ${pe>=50?'<span class=B>Employé</span>':'<span class=R>Non employé</span>'}
 <div class="note">Hypothèse « naïve » : X et Y indépendants sachant la classe. Les zones colorées montrent la frontière gaussienne (déplacez ★). Dans le projet, Naive Bayes atteint ≈ 85 % (la règle « ≥ 3 technologies » n'est pas gaussienne) contre 100 % pour le SVC.</div>`}
T.push(['🧮 Naive Bayes',drawNB,'Régions de décision gaussiennes',1]);
