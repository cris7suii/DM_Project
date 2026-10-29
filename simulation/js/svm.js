/* ---------- SVM linéaire ---------- */
let w=[1,0],bb=-2.5,C=1,anim=null;
function drawSVM(){base();const n=Math.hypot(...w)||1e-6,M=2/n;
 const line=(o,st,dash)=>{g.strokeStyle=st;g.setLineDash(dash?[5,4]:[]);g.beginPath();let a=[],b=[];
  for(const x of[0,10]){a.push([x,(o-bb-w[0]*x)/(w[1]||1e-6)])}
  if(Math.abs(w[1])<1e-3){const x=(o-bb)/w[0];g.moveTo(sx(x),sy(0));g.lineTo(sx(x),sy(10))}else{g.moveTo(sx(0),sy(a[0][1]));g.lineTo(sx(10),sy(a[1][1]))}g.stroke();g.setLineDash([])};
 line(0,'#fbbf24');line(1,'#38bdf8',1);line(-1,'#f43f5e',1);dots();
 const f=p=>w[0]*p.x+w[1]*p.y+bb;let ok=pts.filter(p=>(f(p)>0?1:0)==p.c).length,hinge=pts.reduce((s,p)=>s+Math.max(0,1-(p.c?1:-1)*f(p)),0);
 $('#panel').innerHTML=`<h2>Linear SVC — hyperplan wᵀx + b = 0 (le champion du projet)</h2>
 <div class="row">w₁ <input type=range min=-3 max=3 step=.05 value=${w[0]} oninput="w[0]=+this.value;drawSVM()"><b>${w[0].toFixed(2)}</b></div>
 <div class="row">w₂ <input type=range min=-3 max=3 step=.05 value=${w[1]} oninput="w[1]=+this.value;drawSVM()"><b>${w[1].toFixed(2)}</b></div>
 <div class="row">b &nbsp;<input type=range min=-12 max=12 step=.1 value=${bb} oninput="bb=+this.value;drawSVM()"><b>${bb.toFixed(2)}</b></div>
 <div class="row">C <input type=range min=.01 max=10 step=.01 value=${C} oninput="C=+this.value;drawSVM()"><b>${C}</b>
 <button class="btn" onclick="train()">▶ Entraîner (descente de sous-gradient)</button></div>
 Marge M = 2/‖w‖ = <b>${M.toFixed(2)}</b> · Exactitude = <b>${(ok/pts.length*100).toFixed(1)} %</b><br>
 Objectif ½‖w‖² + C·Σ hinge = <b>${(.5*n*n+C*hinge).toFixed(2)}</b>
 <div class="note">Sur la règle exacte, le SVC trouve w≈(1,0), b≈−2.5 : la même règle « X ≥ 3 » (score 100 %). En bruit, aucune droite ne sépare : la marge souple (C) tolère les erreurs.</div>`}
function train(){clearInterval(anim);w=[.1,.1];bb=0;let it=0;anim=setInterval(()=>{const lr=.02/(1+it/50);let gw=[w[0],w[1]],gb=0;
 pts.forEach(p=>{const y=p.c?1:-1;if(y*(w[0]*p.x+w[1]*p.y+bb)<1){gw[0]-=C*y*p.x/pts.length*10;gw[1]-=C*y*p.y/pts.length*10;gb-=C*y/pts.length*10}});
 w[0]-=lr*gw[0]*5;w[1]-=lr*gw[1]*5;bb-=lr*gb*5;drawSVM();if(++it>150)clearInterval(anim)},30)}
T.push(['📐 SVM linéaire',drawSVM,'Hyperplan & marge',0]);
