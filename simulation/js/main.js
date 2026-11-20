/* ---------- app ---------- */
$('#nav').innerHTML=T.map((t,i)=>`<button id="t${i}" onclick="go(${i})">${t[0]}</button>`).join('');
function go(i){clearInterval(anim);tab=i;T.forEach((_,j)=>$('#t'+j).classList.toggle('on',i==j));$('#ct').textContent=T[i][2];T[i][1]()}
function setD(n){noisy=n;$('#d1').classList.toggle('on',!n);$('#d2').classList.toggle('on',n);seed=7;gen();ns=1;go(tab)}
$('#d1').onclick=()=>setD(false);$('#d2').onclick=()=>setD(true);$('#d3').onclick=()=>{seed=Math.floor(Math.random()*9999)+1;gen();ns=1;go(tab)};
function mv(e){const r=cv.getBoundingClientRect(),k=W/r.width;Q.x=Math.min(10,Math.max(0,ix((e.clientX-r.left)*k)));Q.y=Math.min(10,Math.max(0,iy((e.clientY-r.top)*k)));if(T[tab]&&T[tab][3])T[tab][1]()}
let dn=0;cv.onmousedown=e=>{dn=1;mv(e)};cv.onmousemove=e=>dn&&mv(e);window.onmouseup=()=>dn=0;
cv.ontouchmove=e=>{e.preventDefault();mv(e.touches[0])};
gen();go(0);
/* ---------- accessibilité (phase 4) : ajouts uniquement ---------- */
(function(){
  /* 1) aria-pressed : suit la classe "on" des boutons */
  const btns=[...document.querySelectorAll('#nav button,#d1,#d2')];
  const sync=b=>b.setAttribute('aria-pressed',b.classList.contains('on')?'true':'false');
  const obs=new MutationObserver(ms=>ms.forEach(m=>sync(m.target)));
  btns.forEach(b=>{sync(b);obs.observe(b,{attributes:true,attributeFilter:['class']})});

  /* 2) Clavier : flèches pour déplacer ★ (Maj = pas plus grand) */
  cv.setAttribute('tabindex','0');
  cv.setAttribute('role','application');
  cv.setAttribute('aria-label','Plan 2D : utilisez les flèches du clavier pour déplacer le point ★');
  cv.addEventListener('focus',()=>{cv.style.outline='2px solid #818cf8';cv.style.outlineOffset='2px'});
  cv.addEventListener('blur',()=>{cv.style.outline=''});
  cv.addEventListener('keydown',e=>{
    const s=e.shiftKey?1:0.2;
    let dx=0,dy=0;
    if(e.key==='ArrowLeft')dx=-s;
    else if(e.key==='ArrowRight')dx=s;
    else if(e.key==='ArrowUp')dy=s;
    else if(e.key==='ArrowDown')dy=-s;
    else return;
    e.preventDefault();
    Q.x=Math.min(10,Math.max(0,Q.x+dx));
    Q.y=Math.min(10,Math.max(0,Q.y+dy));
    if(T[tab]&&T[tab][3])T[tab][1]();
  });

  /* 3) Tactile : un simple toucher place ★ (avant, seul le glissement marchait) */
  cv.addEventListener('touchstart',e=>{e.preventDefault();mv(e.touches[0])},{passive:false});

  /* 4) Restaurer le focus des curseurs après reconstruction du panneau */
  let last=null;
  document.addEventListener('input',e=>{
    const t=e.target;
    if(t&&t.type==='range'){
      const all=[...document.querySelectorAll('input[type=range]')];
      last={id:t.id,i:all.indexOf(t),time:Date.now()};
    }
  },true);
  document.addEventListener('input',()=>{
    setTimeout(()=>{
      if(!last||Date.now()-last.time>500)return;
      if(document.activeElement&&document.activeElement!==document.body)return;
      const all=[...document.querySelectorAll('input[type=range]')];
      const el=(last.id&&document.getElementById(last.id))||all[last.i];
      if(el)el.focus({preventScroll:true});
    },0);
  });
})();