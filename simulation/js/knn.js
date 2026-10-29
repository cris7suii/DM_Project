/* ---------- KNN ---------- */
let K=5,wt=0;
function knn(){const d=pts.map(p=>({p,d:Math.hypot(p.x-Q.x,p.y-Q.y)})).sort((a,b)=>a.d-b.d).slice(0,K);
 let v=[0,0];d.forEach(o=>v[o.p.c]+=wt?1/(o.d+1e-6):1);return{d,v}}
function drawKNN(){base();const{d,v}=knn();g.strokeStyle='#fbbf24';g.setLineDash([4,4]);
 d.forEach(o=>{g.beginPath();g.moveTo(sx(Q.x),sy(Q.y));g.lineTo(sx(o.p.x),sy(o.p.y));g.stroke()});g.setLineDash([]);
 g.beginPath();g.arc(sx(Q.x),sy(Q.y),d[d.length-1].d/10*(W-2*P),0,7);g.stroke();dots();star();
 const t=v[0]+v[1],pe=v[1]/t*100;
 $('#panel').innerHTML=`<h2>Vote des K plus proches voisins</h2>
 <div class="row">K=<b>${K}</b><input type=range min=1 max=15 value=${K} oninput="K=+this.value;drawKNN()">
 <button class="btn ${wt?'on':''}" onclick="wt=1-wt;drawKNN()">Pondéré 1/d</button></div>
 <div class="bar"><i style="width:${pe}%;background:var(--b)"></i><i style="width:${100-pe}%;background:var(--r)"></i></div>
 <b>${pe>=50?'<span class=B>Employé</span>':'<span class=R>Non employé</span>'}</b> — P(employé)=${pe.toFixed(1)} %
 <table><tr><th>#</th><th>(X,Y)</th><th>Classe</th><th>d</th></tr>${d.map((o,i)=>`<tr><td>${i+1}</td><td>(${o.p.x},${o.p.y})</td><td class="${o.p.c?'B':'R'}">${o.p.c?'Employé':'Non'}</td><td>${o.d.toFixed(2)}</td></tr>`).join('')}</table>
 <div class="note">Cliquez/glissez pour déplacer ★. Distance euclidienne ; sur données bruitées (profil seul) le KNN reste proche de 57 %.</div>`}
T.push(['🎯 KNN',drawKNN,'Plan 2D — voisins',1]);
