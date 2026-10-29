/* ---------- app ---------- */
$('#nav').innerHTML=T.map((t,i)=>`<button id="t${i}" onclick="go(${i})">${t[0]}</button>`).join('');
function go(i){clearInterval(anim);tab=i;T.forEach((_,j)=>$('#t'+j).classList.toggle('on',i==j));$('#ct').textContent=T[i][2];T[i][1]()}
function setD(n){noisy=n;$('#d1').classList.toggle('on',!n);$('#d2').classList.toggle('on',n);seed=7;gen();ns=1;go(tab)}
$('#d1').onclick=()=>setD(false);$('#d2').onclick=()=>setD(true);$('#d3').onclick=()=>{seed=Math.floor(Math.random()*9999)+1;gen();ns=1;go(tab)};
function mv(e){const r=cv.getBoundingClientRect(),k=W/r.width;Q.x=Math.min(10,Math.max(0,ix((e.clientX-r.left)*k)));Q.y=Math.min(10,Math.max(0,iy((e.clientY-r.top)*k)));if(T[tab]&&T[tab][3])T[tab][1]()}
let dn=0;cv.onmousedown=e=>{dn=1;mv(e)};cv.onmousemove=e=>dn&&mv(e);window.onmouseup=()=>dn=0;
cv.ontouchmove=e=>{e.preventDefault();mv(e.touches[0])};
gen();go(0);
