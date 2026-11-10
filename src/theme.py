"""Thème EmployML (design Figma) pour l'app Streamlit.

Usage dans app.py :
    from theme import inject_theme, page_header, hero, metric_cards, warning_card
    st.set_page_config(page_title="EmployML", page_icon="💼", layout="wide")
    inject_theme()
"""
from __future__ import annotations

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root{
  --bg:#0f172a; --bg-deep:#0b1120; --surface:#182438; --surface-2:#1e293b;
  --line:rgba(148,163,184,.14); --text:#f1f5f9; --muted:#94a3b8; --soft:#cbd5e1;
  --indigo:#6366f1; --violet:#7c3aed; --cyan:#22d3ee;
  --green:#22c55e; --amber:#f59e0b; --red:#ef4444;
}
html, body, [class*="st-"], .stApp { font-family:'Inter',sans-serif; }
.stApp{
  background: radial-gradient(circle at 80% 8%, rgba(99,102,241,.10), transparent 27%), var(--bg);
  color: var(--text);
}
/* header Streamlit + marges */
header[data-testid="stHeader"]{ background:transparent; }
.block-container{ max-width:1280px; padding-top:2.2rem; padding-bottom:3rem; }
#MainMenu, footer{ visibility:hidden; }

/* Onglets = barre de navigation */
div[data-testid="stTabs"] [role="tablist"]{
  gap:4px; border-bottom:1px solid var(--line); margin-bottom:1.6rem;
}
div[data-testid="stTabs"] button[role="tab"]{
  color:var(--muted); font-size:.85rem; font-weight:500; padding:.7rem 1rem;
  background:transparent; border-radius:8px 8px 0 0;
}
div[data-testid="stTabs"] button[role="tab"]:hover{ color:var(--text); }
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{ color:var(--text); }
div[data-testid="stTabs"] [data-baseweb="tab-highlight"]{ background:var(--indigo); height:2px; }

/* Titres */
h1,h2,h3{ letter-spacing:-.02em; color:var(--text); }
.eyebrow{ display:flex; align-items:center; gap:8px; color:var(--cyan);
  font-size:.7rem; font-weight:600; letter-spacing:.16em; text-transform:uppercase; }
.eyebrow span{ width:18px; height:2px; background:var(--cyan); display:inline-block; }
.page-title{ font-size:2.1rem; font-weight:700; letter-spacing:-.03em; margin:.4rem 0 .3rem; }
.page-desc{ color:var(--muted); font-size:.98rem; margin-bottom:1.4rem; max-width:640px; }
.hero-title{ font-size:3rem; line-height:1.1; font-weight:700; letter-spacing:-.04em; margin:.6rem 0 1rem; }
.hero-title em{ font-style:normal;
  background:linear-gradient(90deg,var(--indigo),var(--cyan));
  -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero-copy{ color:var(--muted); font-size:1.05rem; max-width:520px; }

/* Cartes */
.card{ background:var(--surface); border:1px solid var(--line); border-radius:16px;
  padding:1.3rem 1.4rem; box-shadow:0 10px 30px rgba(0,0,0,.18); height:100%; }
.card-kicker{ color:var(--muted); font-size:.68rem; font-weight:600; letter-spacing:.14em; text-transform:uppercase; }
.metric-card{ display:flex; gap:14px; align-items:center; }
.metric-icon{ width:44px; height:44px; border-radius:12px; display:grid; place-items:center; font-size:1.2rem; }
.metric-icon.indigo{ background:rgba(99,102,241,.16); }
.metric-icon.cyan{ background:rgba(34,211,238,.14); }
.metric-icon.violet{ background:rgba(124,58,237,.16); }
.metric-icon.green{ background:rgba(34,197,94,.14); }
.metric-value{ font-size:1.7rem; font-weight:700; letter-spacing:-.03em; line-height:1.1; }
.metric-label{ color:var(--soft); font-size:.85rem; font-weight:500; }
.metric-hint{ color:var(--muted); font-size:.72rem; }

/* Bannière d'avertissement */
.warning-card{ display:flex; gap:14px; background:rgba(245,158,11,.08);
  border:1px solid rgba(245,158,11,.35); border-radius:16px; padding:1.2rem 1.4rem; }
.warning-card .w-icon{ font-size:1.4rem; }
.warning-card h3{ margin:.2rem 0 .3rem; font-size:1.05rem; }
.warning-card p{ margin:0; color:var(--soft); font-size:.9rem; }

/* Badges */
.badge{ display:inline-block; padding:.2rem .65rem; border-radius:999px; font-size:.72rem; font-weight:600;
  background:rgba(148,163,184,.14); color:var(--soft); }
.badge.champion{ background:rgba(99,102,241,.2); color:#a5b4fc; }
.badge.ok{ background:rgba(34,197,94,.16); color:#4ade80; }
.badge.warn{ background:rgba(245,158,11,.16); color:#fbbf24; }
.badge.bad{ background:rgba(239,68,68,.16); color:#f87171; }

/* Boutons Streamlit */
.stButton>button, .stFormSubmitButton>button{
  background:linear-gradient(145deg,var(--violet),#4f46e5); color:#fff; border:0;
  border-radius:10px; padding:.55rem 1.2rem; font-weight:600;
  box-shadow:0 8px 24px rgba(99,102,241,.3); }
.stButton>button:hover{ filter:brightness(1.1); color:#fff; }

/* Métriques natives */
div[data-testid="stMetric"]{ background:var(--surface); border:1px solid var(--line);
  border-radius:14px; padding:1rem 1.2rem; }
div[data-testid="stMetricLabel"]{ color:var(--muted); }

/* Tableaux / dataframes */
div[data-testid="stDataFrame"]{ border:1px solid var(--line); border-radius:12px; overflow:hidden; }

/* Inputs */
div[data-baseweb="select"]>div, .stTextInput input, .stNumberInput input{
  background:var(--surface-2); border-color:var(--line); border-radius:10px; }
.stMultiSelect span[data-baseweb="tag"]{ background:rgba(99,102,241,.25); border-radius:999px; }

/* Pipeline */
.pipeline{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-top:1rem; }
.pipe-step{ background:var(--surface-2); border:1px solid var(--line); border-radius:12px; padding:.9rem 1rem; }
.pipe-step b{ display:inline-grid; place-items:center; width:26px; height:26px; border-radius:50%;
  background:rgba(34,197,94,.16); color:#4ade80; font-size:.75rem; margin-bottom:.5rem; }
.pipe-step strong{ display:block; font-size:.9rem; }
.pipe-step small{ color:var(--muted); }

/* Résultat de prédiction */
.result-card{ text-align:center; }
.result-badge{ display:inline-flex; align-items:center; gap:8px; padding:.6rem 1.3rem; border-radius:999px;
  font-weight:700; font-size:1.1rem; margin:.8rem 0; }
.result-badge.positive{ background:rgba(34,197,94,.16); color:#4ade80; border:1px solid rgba(34,197,94,.4); }
.result-badge.negative{ background:rgba(245,158,11,.14); color:#fbbf24; border:1px solid rgba(245,158,11,.4); }
.gauge{ --score:0deg; width:150px; height:150px; border-radius:50%; margin:1rem auto 0;
  background:conic-gradient(var(--indigo) var(--score), rgba(148,163,184,.18) 0);
  display:grid; place-items:center; }
.gauge div{ width:116px; height:116px; border-radius:50%; background:var(--surface); display:grid; place-content:center; }
.gauge strong{ font-size:1.6rem; }
.gauge span{ color:var(--muted); font-size:.7rem; }
.result-note{ color:var(--muted); font-size:.85rem; }

/* Disclaimer éthique */
.ethical-card{ display:flex; gap:14px; background:rgba(99,102,241,.08);
  border:1px solid rgba(99,102,241,.35); border-radius:16px; padding:1.2rem 1.4rem; margin-bottom:1.2rem; }
.ethical-card h3{ margin:.2rem 0 .3rem; font-size:1.05rem; }
.ethical-card p{ margin:0; color:var(--soft); font-size:.9rem; }
@media (max-width:900px){ .pipeline{ grid-template-columns:1fr 1fr; } .hero-title{ font-size:2.1rem; } }
</style>
"""


def inject_theme() -> None:
    """À appeler une seule fois, juste après st.set_page_config()."""
    st.markdown(CSS, unsafe_allow_html=True)


def _html(s: str) -> None:
    # supprime l'indentation pour éviter que Markdown ne transforme le HTML en bloc de code
    st.markdown("".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, description: str = "") -> None:
    _html(f'<div class="eyebrow"><span></span>{eyebrow}</div>'
          f'<div class="page-title">{title}</div>'
          f'<div class="page-desc">{description}</div>')


def hero(eyebrow: str, line1: str, line2: str, copy: str) -> None:
    _html(f'<div class="eyebrow"><span></span>{eyebrow}</div>'
          f'<div class="hero-title">{line1}<br><em>{line2}</em></div>'
          f'<div class="hero-copy">{copy}</div>')


def metric_cards(items: list[dict]) -> None:
    """items = [{"icon":"🗄️","value":"73 462","label":"Candidats","hint":"...","tone":"indigo"}, ...]"""
    cols = st.columns(len(items))
    for col, m in zip(cols, items):
        with col:
            _html(f'<div class="card metric-card">'
                  f'<div class="metric-icon {m.get("tone", "indigo")}">{m["icon"]}</div>'
                  f'<div><div class="metric-value">{m["value"]}</div>'
                  f'<div class="metric-label">{m["label"]}</div>'
                  f'<div class="metric-hint">{m.get("hint", "")}</div></div></div>')


def warning_card(title: str, text: str, icon: str = "⚠️") -> None:
    _html(f'<div class="warning-card"><div class="w-icon">{icon}</div>'
          f'<div><div class="card-kicker">Contexte important</div>'
          f'<h3>{title}</h3><p>{text}</p></div></div>')


def badge(text: str, kind: str = "") -> str:
    """Retourne le HTML d'un badge (kind : champion | ok | warn | bad)."""
    return f'<span class="badge {kind}">{text}</span>'


def pipeline(steps: list[tuple[str, str]]) -> None:
    """steps = [("Nettoyage", "Valeurs manquantes"), ...]"""
    inner = "".join(f'<div class="pipe-step"><b>{i + 1}</b><strong>{t}</strong><small>{d}</small></div>'
                    for i, (t, d) in enumerate(steps))
    _html(f'<div class="card"><div class="card-kicker">Workflow</div>'
          f'<div class="pipeline">{inner}</div></div>')


def result_card(pred: int, score_text: str = "", proba: float | None = None) -> None:
    gauge = ""
    if proba is not None:
        gauge = (f'<div class="gauge" style="--score:{proba * 360:.0f}deg"><div>'
                 f'<strong>{proba:.0%}</strong><span>probabilité</span></div></div>')
    kind, label = ("positive", "✅ Employé (1)") if pred == 1 else ("negative", "⚠️ Non employé (0)")
    _html(f'<div class="card result-card"><div class="card-kicker">Résultat de la prédiction</div>'
          f'{gauge}<div class="result-badge {kind}">{label}</div>'
          f'<div class="result-note">{score_text}</div></div>')


def ethical_card(title: str, text: str, icon: str = "🛡️") -> None:
    _html(f'<div class="ethical-card"><div class="w-icon">{icon}</div>'
          f'<div><div class="card-kicker">Avertissement éthique</div><h3>{title}</h3><p>{text}</p></div></div>')
