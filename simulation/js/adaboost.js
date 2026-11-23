/* ---------- AdaBoost (souches de décision) ---------- */
let abR = 5, abCache = null;

/* Entraîne R souches successives ; chaque tour re-pondère les points mal classés */
function abFit(R) {
  if (abCache && abCache.pts === pts && abCache.R >= R) return abCache.models.slice(0, R);
  const n = pts.length, y = pts.map(p => (p.c ? 1 : -1));
  let wts = Array(n).fill(1 / n);
  const models = [];
  for (let r = 0; r < R; r++) {
    let best = null;
    for (const f of ['x', 'y'])
      for (let t = .5; t < 10; t += .5)
        for (const s of [1, -1]) {
          let err = 0;
          pts.forEach((p, i) => { if ((s * (p[f] > t ? 1 : -1)) !== y[i]) err += wts[i]; });
          if (!best || err < best.err) best = { f, t, s, err };
        }
    const eps = Math.max(best.err, 1e-6);
    const alpha = .5 * Math.log((1 - eps) / eps);
    models.push({ ...best, alpha, w: wts.slice() });          // w = poids AVANT ce tour
    wts = wts.map((v, i) => v * Math.exp(-alpha * y[i] * best.s * (pts[i][best.f] > best.t ? 1 : -1)));
    const z = wts.reduce((a, b) => a + b, 0);
    wts = wts.map(v => v / z);
    models[r].wNext = wts.slice();                             // poids APRÈS ce tour
  }
  abCache = { pts, R, models };
  return models;
}
const abH = (m, x, y) => m.s * ((m.f === 'x' ? x : y) > m.t ? 1 : -1);
const abScore = (ms, x, y) => ms.reduce((a, m) => a + m.alpha * abH(m, x, y), 0);

function drawAB() {
  base();
  const ms = abFit(abR), N = 28, s = (W - 2 * P) / N;
  const tot = ms.reduce((a, m) => a + m.alpha, 0) || 1;
  /* zones de décision de l'ensemble */
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    const sc = abScore(ms, (i + .5) / N * 10, (j + .5) / N * 10);
    g.fillStyle = `rgba(${sc > 0 ? '16,185,129' : '249,115,22'},${Math.min(.35, .08 + Math.abs(sc) / tot * .3)})`;
    g.fillRect(sx(i / N * 10), sy((j + 1) / N * 10), s + 1, s + 1);
  }
  /* souches (la dernière en surbrillance) */
  ms.forEach((m, i) => {
    g.strokeStyle = i === ms.length - 1 ? '#22d3ee' : 'rgba(129,140,248,.55)';
    g.lineWidth = i === ms.length - 1 ? 2.5 : 1.2; g.beginPath();
    m.f === 'x' ? (g.moveTo(sx(m.t), sy(0)), g.lineTo(sx(m.t), sy(10))) : (g.moveTo(sx(0), sy(m.t)), g.lineTo(sx(10), sy(m.t)));
    g.stroke();
  });
  g.lineWidth = 1;
  /* points : taille ∝ poids avant le dernier tour */
  const last = ms[ms.length - 1], n = pts.length;
  pts.forEach((p, i) => {
    const r = Math.min(13, 4 + last.w[i] * n * 2.2);
    g.fillStyle = col(p.c); g.beginPath();
    p.c ? g.arc(sx(p.x), sy(p.y), r, 0, 7) : g.rect(sx(p.x) - r, sy(p.y) - r, 2 * r, 2 * r);
    g.fill();
  });
  star();
  const ok = pts.filter(p => (abScore(ms, p.x, p.y) > 0 ? 1 : 0) === p.c).length;
  const sc = abScore(ms, Q.x, Q.y), pe = 100 / (1 + Math.exp(-2 * sc)), f2 = v => v.toFixed(2);
  const rows = ms.map((m, i) => `<tr><td>${i + 1}</td><td>${m.f === 'x' ? 'X' : 'Y'} ${m.s > 0 ? '>' : '≤'} ${m.t}</td><td>${m.err.toFixed(3)}</td><td>${f2(m.alpha)}</td></tr>`).slice(-8).join('');
  $('#panel').innerHTML = `<h2>AdaBoost — H(x) = signe( Σ αₜ·hₜ(x) )</h2>
 <div class="row">Tours T=<b>${abR}</b><input type=range min=1 max=30 value=${abR} oninput="abR=+this.value;drawAB()">
 <button class="btn" onclick="abR=Math.min(30,abR+1);drawAB()">+1 tour ▶</button></div>
 <div class="bar"><i style="width:${pe}%;background:var(--b)"></i><i style="width:${100 - pe}%;background:var(--r)"></i></div>
 <b>${sc > 0 ? '<span class=B>Employé</span>' : '<span class=R>Non employé</span>'}</b> — score ★ = ${f2(sc)} · exactitude d'apprentissage = <b>${(ok / n * 100).toFixed(1)} %</b>
 <table><tr><th>Tour</th><th>Souche hₜ</th><th>Erreur εₜ</th><th>Poids αₜ</th></tr>${rows}</table>
 <div class="note">αₜ = ½·ln((1−εₜ)/εₜ). Les points mal classés grossissent (poids ↑) pour forcer la souche suivante à s'y intéresser. Déplacez ★ pour voir le vote pondéré. Sur la règle exacte, la 1ʳᵉ souche <code>X > 2.5</code> a ε≈0 : AdaBoost converge dès le tour 1. En profil seul (bruit), chaque souche reste proche de ε≈0,4–0,5 : si l'exactitude d'apprentissage monte avec T, c'est du sur-apprentissage (à valider sur un jeu de test).</div>`;
}
T.push(['⚡ AdaBoost', drawAB, 'Boosting de souches — poids des points', 1]);
