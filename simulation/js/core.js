const $=s=>document.querySelector(s),cv=$('#cv'),g=cv.getContext('2d'),W=560,P=30;
let seed=7,rnd=()=>(seed=(seed*16807)%2147483647)/2147483647,noisy=false,tab=0,pts=[],Q={x:4,y:6};
const sx=v=>P+v/10*(W-2*P),sy=v=>W-P-v/10*(W-2*P),ix=p=>(p-P)/(W-2*P)*10,iy=p=>(W-P-p)/(W-2*P)*10;
function gen(){pts=[];for(let i=0;i<60;i++){const x=Math.round(rnd()*100)/10,y=Math.round(rnd()*100)/10;
 pts.push({x,y,c:noisy?(rnd()<.57?1:0):(Math.floor(x)>=3?1:0)})}}
const col=c=>c?'#10b981':'#f97316';
function base(){g.clearRect(0,0,W,W);g.strokeStyle='#273549';g.fillStyle='#94a3b8';g.font='11px sans-serif';
 for(let i=0;i<=10;i+=2){g.beginPath();g.moveTo(sx(i),sy(0));g.lineTo(sx(i),sy(10));g.moveTo(sx(0),sy(i));g.lineTo(sx(10),sy(i));g.stroke();g.fillText(i,sx(i)-3,W-10);g.fillText(i,8,sy(i)+4)}}
function dots(){pts.forEach(p=>{g.fillStyle=col(p.c);g.beginPath();if(p.c)g.arc(sx(p.x),sy(p.y),6,0,7);else g.rect(sx(p.x)-5,sy(p.y)-5,10,10);g.fill()})}
function star(){g.fillStyle='#818cf8';g.font='22px sans-serif';g.fillText('◆',sx(Q.x)-8,sy(Q.y)+7)}
const gini=a=>{const n=a.length;if(!n)return 0;const p=a.filter(t=>t.c).length/n;return 1-p*p-(1-p)*(1-p)};
const T=[];   // registre des onglets : chaque module fait T.push([label, fn, titre, ★ déplaçable])
