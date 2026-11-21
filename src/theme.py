"""Thème EmployML (Design System Moderne & Pédagogique) pour Streamlit.

Offre une interface de niveau Enterprise AI / SaaS :
- Mode sombre profond avec éclairages ambiants subtils (mesh radial gradients).
- Cartes en glassmorphism avec bordures lumineuses et micro-interactions au survol.
- Typographie soignée (Inter / Plus Jakarta Sans) et hiérarchie visuelle contrastée.
- Composants métier : KPIs interactifs, pipeline visuel, cartes de décision et jauges.

Usage dans app.py :
    from theme import inject_theme, page_header, hero, metric_cards, warning_card, result_card
    st.set_page_config(page_title="EmployML", page_icon="💼", layout="wide")
    inject_theme()
"""
from __future__ import annotations

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

:root {
  --bg: #090d16;
  --bg-deep: #060911;
  --surface: rgba(18, 26, 43, 0.75);
  --surface-hover: rgba(26, 37, 60, 0.85);
  --surface-2: #161f33;
  --surface-3: #1e293b;
  --line: rgba(148, 163, 184, 0.12);
  --line-strong: rgba(148, 163, 184, 0.22);
  
  --text: #f8fafc;
  --muted: #94a3b8;
  --soft: #cbd5e1;
  
  --indigo: #6366f1;
  --indigo-glow: rgba(99, 102, 241, 0.35);
  --violet: #8b5cf6;
  --cyan: #06b6d4;
  --cyan-glow: rgba(6, 182, 212, 0.3);
  --green: #10b981;
  --green-glow: rgba(16, 185, 129, 0.25);
  --amber: #f59e0b;
  --amber-glow: rgba(245, 158, 11, 0.25);
  --red: #f43f5e;
  --red-glow: rgba(244, 63, 94, 0.25);
}

html, body, [class*="st-"], .stApp {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  color: var(--text);
}

h1, h2, h3, .hero-title, .page-title {
  font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
}

code, kbd, samp, pre {
  font-family: 'Fira Code', monospace;
}

/* Fond immersif avec éclairages d'ambiance */
.stApp {
  background:
    radial-gradient(circle at 12% 8%, rgba(99, 102, 241, 0.14), transparent 32%),
    radial-gradient(circle at 88% 14%, rgba(6, 182, 212, 0.12), transparent 30%),
    radial-gradient(circle at 50% 95%, rgba(139, 92, 246, 0.08), transparent 45%),
    var(--bg);
  background-attachment: fixed;
  color: var(--text);
}

/* Header Streamlit + conteneur principal */
header[data-testid="stHeader"] {
  background: transparent !important;
}
.block-container {
  max-width: 1320px !important;
  padding-top: 1.8rem !important;
  padding-bottom: 3.5rem !important;
}
#MainMenu, footer {
  visibility: hidden;
}

/* Navigation par onglets (Pills modernes) */
div[data-testid="stTabs"] [role="tablist"] {
  gap: 8px;
  border-bottom: 1px solid var(--line);
  padding: 4px 2px 10px 2px;
  margin-bottom: 2rem;
  overflow-x: auto;
}
div[data-testid="stTabs"] button[role="tab"] {
  color: var(--muted);
  font-size: 0.88rem;
  font-weight: 500;
  padding: 0.65rem 1.15rem;
  background: rgba(30, 41, 59, 0.4);
  border: 1px solid var(--line);
  border-radius: 12px;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}
div[data-testid="stTabs"] button[role="tab"]:hover {
  color: var(--text);
  background: rgba(30, 41, 59, 0.8);
  border-color: rgba(99, 102, 241, 0.35);
  transform: translateY(-1px);
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  color: #ffffff !important;
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.28), rgba(139, 92, 246, 0.18)) !important;
  border-color: rgba(99, 102, 241, 0.6) !important;
  box-shadow: 0 4px 20px rgba(99, 102, 241, 0.25);
  font-weight: 600;
}
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
  display: none;
}

/* Typographie & En-têtes */
.eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: var(--cyan);
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  margin-bottom: 0.3rem;
  background: rgba(6, 182, 212, 0.1);
  padding: 0.25rem 0.75rem;
  border-radius: 999px;
  border: 1px solid rgba(6, 182, 212, 0.25);
}
.eyebrow span {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--cyan);
  display: inline-block;
  box-shadow: 0 0 8px var(--cyan);
}
.page-title {
  font-size: 2.2rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  margin: 0.5rem 0 0.4rem;
  color: #ffffff;
}
.page-desc {
  color: var(--muted);
  font-size: 1rem;
  line-height: 1.55;
  margin-bottom: 1.6rem;
  max-width: 680px;
}
.hero-title {
  font-size: 3.1rem;
  line-height: 1.12;
  font-weight: 800;
  letter-spacing: -0.04em;
  margin: 0.7rem 0 1.1rem;
  color: #ffffff;
}
.hero-title em {
  font-style: normal;
  background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 50%, #22d3ee 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.hero-copy {
  color: var(--soft);
  font-size: 1.05rem;
  line-height: 1.6;
  max-width: 580px;
}

/* Cartes en glassmorphism */
.card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 1.4rem 1.5rem;
  box-shadow: 0 12px 32px -8px rgba(0, 0, 0, 0.35);
  backdrop-filter: blur(16px);
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
  height: 100%;
}
.card:hover {
  border-color: var(--line-strong);
  box-shadow: 0 16px 40px -8px rgba(0, 0, 0, 0.45);
}
.card-kicker {
  color: var(--muted);
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  margin-bottom: 0.5rem;
}

/* Cartes métriques (KPI) */
.metric-card {
  display: flex;
  gap: 16px;
  align-items: center;
}
.metric-card:hover {
  transform: translateY(-2px);
  border-color: rgba(99, 102, 241, 0.3);
}
.metric-icon {
  width: 48px;
  height: 48px;
  border-radius: 14px;
  display: grid;
  place-items: center;
  font-size: 1.35rem;
  flex-shrink: 0;
  border: 1px solid transparent;
}
.metric-icon.indigo {
  background: rgba(99, 102, 241, 0.18);
  border-color: rgba(99, 102, 241, 0.3);
  box-shadow: 0 0 18px rgba(99, 102, 241, 0.2);
}
.metric-icon.cyan {
  background: rgba(6, 182, 212, 0.16);
  border-color: rgba(6, 182, 212, 0.3);
  box-shadow: 0 0 18px rgba(6, 182, 212, 0.2);
}
.metric-icon.violet {
  background: rgba(139, 92, 246, 0.18);
  border-color: rgba(139, 92, 246, 0.3);
  box-shadow: 0 0 18px rgba(139, 92, 246, 0.2);
}
.metric-icon.green {
  background: rgba(16, 185, 129, 0.16);
  border-color: rgba(16, 185, 129, 0.3);
  box-shadow: 0 0 18px rgba(16, 185, 129, 0.2);
}
.metric-value {
  font-size: 1.8rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  line-height: 1.15;
  color: #ffffff;
}
.metric-label {
  color: var(--soft);
  font-size: 0.88rem;
  font-weight: 600;
  margin-top: 0.1rem;
}
.metric-hint {
  color: var(--muted);
  font-size: 0.74rem;
  margin-top: 0.2rem;
}

/* Bannières de Contexte & Alertes Pédagogiques */
.warning-card {
  display: flex;
  gap: 16px;
  background: linear-gradient(135deg, rgba(245, 158, 11, 0.09), rgba(245, 158, 11, 0.04));
  border: 1px solid rgba(245, 158, 11, 0.35);
  border-left: 4px solid var(--amber);
  border-radius: 16px;
  padding: 1.3rem 1.5rem;
  box-shadow: 0 10px 25px -5px rgba(245, 158, 11, 0.1);
}
.warning-card .w-icon {
  font-size: 1.6rem;
  flex-shrink: 0;
}
.warning-card h3 {
  margin: 0.1rem 0 0.35rem;
  font-size: 1.1rem;
  font-weight: 700;
  color: #fde68a;
}
.warning-card p {
  margin: 0;
  color: #fef3c7;
  font-size: 0.92rem;
  line-height: 1.5;
}

/* Badges */
.badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 0.3rem 0.8rem;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 600;
  background: rgba(148, 163, 184, 0.14);
  color: var(--soft);
  border: 1px solid var(--line);
  transition: all 0.2s ease;
}
.badge.champion {
  background: rgba(99, 102, 241, 0.22);
  color: #c7d2fe;
  border-color: rgba(99, 102, 241, 0.45);
  box-shadow: 0 0 12px rgba(99, 102, 241, 0.25);
}
.badge.ok {
  background: rgba(16, 185, 129, 0.18);
  color: #6ee7b7;
  border-color: rgba(16, 185, 129, 0.4);
}
.badge.warn {
  background: rgba(245, 158, 11, 0.18);
  color: #fcd34d;
  border-color: rgba(245, 158, 11, 0.4);
}
.badge.bad {
  background: rgba(244, 63, 94, 0.18);
  color: #fda4af;
  border-color: rgba(244, 63, 94, 0.4);
}

/* Boutons Streamlit & Formulaires */
.stButton > button, .stFormSubmitButton > button {
  background: linear-gradient(135deg, #6366f1 0%, #4f46e5 50%, #4338ca 100%) !important;
  color: #ffffff !important;
  border: 1px solid rgba(255, 255, 255, 0.2) !important;
  border-radius: 12px !important;
  padding: 0.65rem 1.4rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.01em;
  box-shadow: 0 8px 24px rgba(99, 102, 241, 0.35) !important;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 12px 30px rgba(99, 102, 241, 0.5) !important;
  filter: brightness(1.1);
  color: #ffffff !important;
}
.stButton > button:active, .stFormSubmitButton > button:active {
  transform: translateY(0);
}

/* Inputs, Selects & Multiselect */
div[data-testid="stForm"] {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 1.6rem 1.8rem;
  backdrop-filter: blur(16px);
  box-shadow: 0 16px 36px -10px rgba(0, 0, 0, 0.35);
}
div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {
  background: var(--surface-2) !important;
  border: 1px solid var(--line) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  transition: all 0.2s ease;
}
div[data-baseweb="select"] > div:hover, .stTextInput input:hover, .stNumberInput input:hover {
  border-color: rgba(148, 163, 184, 0.3) !important;
}
div[data-baseweb="select"] > div:focus-within, .stTextInput input:focus, .stNumberInput input:focus {
  border-color: var(--indigo) !important;
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.22) !important;
}
.stMultiSelect span[data-baseweb="tag"] {
  background: rgba(99, 102, 241, 0.25) !important;
  border: 1px solid rgba(99, 102, 241, 0.4) !important;
  border-radius: 999px !important;
  color: #e0e7ff !important;
}

/* Pipeline Didactique (Stepper) */
.pipeline {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
  margin-top: 1rem;
}
.pipe-step {
  background: var(--surface-2);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 1.1rem 1.2rem;
  position: relative;
  transition: all 0.2s ease;
}
.pipe-step:hover {
  border-color: rgba(99, 102, 241, 0.35);
  transform: translateY(-2px);
}
.pipe-step b {
  display: inline-grid;
  place-items: center;
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: rgba(16, 185, 129, 0.18);
  color: #34d399;
  border: 1px solid rgba(16, 185, 129, 0.35);
  font-size: 0.8rem;
  margin-bottom: 0.6rem;
  font-weight: 700;
}
.pipe-step strong {
  display: block;
  font-size: 0.95rem;
  color: #ffffff;
  margin-bottom: 0.25rem;
}
.pipe-step small {
  color: var(--muted);
  font-size: 0.8rem;
  line-height: 1.4;
  display: block;
}

/* Carte de Résultat de Prédiction */
.result-card {
  text-align: center;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 1.8rem 1.6rem;
  box-shadow: 0 16px 36px -10px rgba(0, 0, 0, 0.4);
}
.result-badge {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 0.75rem 1.6rem;
  border-radius: 999px;
  font-weight: 700;
  font-size: 1.15rem;
  margin: 1.1rem 0 0.8rem;
  letter-spacing: -0.01em;
}
.result-badge.positive {
  background: rgba(16, 185, 129, 0.18);
  color: #6ee7b7;
  border: 1px solid rgba(16, 185, 129, 0.45);
  box-shadow: 0 0 24px rgba(16, 185, 129, 0.25);
}
.result-badge.negative {
  background: rgba(245, 158, 11, 0.16);
  color: #fcd34d;
  border: 1px solid rgba(245, 158, 11, 0.45);
  box-shadow: 0 0 24px rgba(245, 158, 11, 0.2);
}

/* Jauge circulaire moderne */
.gauge {
  --score: 0deg;
  width: 156px;
  height: 156px;
  border-radius: 50%;
  margin: 1.2rem auto 0.6rem;
  background: conic-gradient(var(--indigo) var(--score), rgba(148, 163, 184, 0.14) 0);
  display: grid;
  place-items: center;
  box-shadow: 0 0 30px rgba(99, 102, 241, 0.25);
  transition: all 0.5s ease-out;
}
.gauge.positive {
  background: conic-gradient(var(--green) var(--score), rgba(148, 163, 184, 0.14) 0);
  box-shadow: 0 0 30px rgba(16, 185, 129, 0.25);
}
.gauge.negative {
  background: conic-gradient(var(--amber) var(--score), rgba(148, 163, 184, 0.14) 0);
  box-shadow: 0 0 30px rgba(245, 158, 11, 0.2);
}
.gauge div {
  width: 122px;
  height: 122px;
  border-radius: 50%;
  background: #0f172a;
  display: grid;
  place-content: center;
  box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.5);
}
.gauge strong {
  font-size: 1.8rem;
  font-weight: 800;
  color: #ffffff;
  letter-spacing: -0.03em;
}
.gauge span {
  color: var(--muted);
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.result-note {
  color: var(--soft);
  font-size: 0.92rem;
  line-height: 1.5;
  margin-top: 0.4rem;
}

/* Carte de Détection des Technologies Clés */
.key-tech-box {
  background: var(--surface-2);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 1rem 1.2rem;
  margin-top: 1rem;
  text-align: left;
}
.key-tech-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}
.key-tech-header span {
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--soft);
}
.key-tech-count {
  font-size: 0.85rem;
  font-weight: 700;
  padding: 0.2rem 0.6rem;
  border-radius: 999px;
}
.key-tech-count.match {
  background: rgba(16, 185, 129, 0.2);
  color: #6ee7b7;
  border: 1px solid rgba(16, 185, 129, 0.4);
}
.key-tech-count.unmatch {
  background: rgba(245, 158, 11, 0.2);
  color: #fcd34d;
  border: 1px solid rgba(245, 158, 11, 0.4);
}

/* Disclaimer Éthique */
.ethical-card {
  display: flex;
  gap: 16px;
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.09), rgba(139, 92, 246, 0.04));
  border: 1px solid rgba(99, 102, 241, 0.35);
  border-left: 4px solid var(--indigo);
  border-radius: 16px;
  padding: 1.3rem 1.5rem;
  margin-bottom: 1.4rem;
  box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.1);
}
.ethical-card .w-icon {
  font-size: 1.6rem;
  flex-shrink: 0;
}
.ethical-card h3 {
  margin: 0.1rem 0 0.35rem;
  font-size: 1.1rem;
  font-weight: 700;
  color: #e0e7ff;
}
.ethical-card p {
  margin: 0;
  color: #c7d2fe;
  font-size: 0.92rem;
  line-height: 1.5;
}

/* Grille Responsive */
@media (max-width: 900px) {
  .pipeline { grid-template-columns: 1fr 1fr; }
  .hero-title { font-size: 2.2rem; }
}
@media (max-width: 600px) {
  .pipeline { grid-template-columns: 1fr; }
}
</style>
"""


def inject_theme() -> None:
    """À appeler une seule fois, juste après st.set_page_config()."""
    st.markdown(CSS, unsafe_allow_html=True)


def _html(s: str) -> None:
    # supprime l'indentation superflue pour éviter les blocs de code Markdown
    st.markdown("".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, description: str = "") -> None:
    """En-tête de section moderne avec pastille eyebrow et titre contrasté."""
    desc_html = f'<div class="page-desc">{description}</div>' if description else ""
    _html(f'<div class="eyebrow"><span></span>{eyebrow}</div>'
          f'<div class="page-title">{title}</div>'
          f'{desc_html}')


def hero(eyebrow: str, line1: str, line2: str, copy: str) -> None:
    """Bannière Hero d'accueil avec titre en dégradé et typographie soignée."""
    _html(f'<div class="eyebrow"><span></span>{eyebrow}</div>'
          f'<div class="hero-title">{line1}<br><em>{line2}</em></div>'
          f'<div class="hero-copy">{copy}</div>')


def metric_cards(items: list[dict]) -> None:
    """Affiche une rangée de cartes métriques KPI haut de gamme.
    items = [{"icon": "🏆", "value": "Linear SVC", "label": "Champion", "hint": "...", "tone": "violet"}, ...]
    """
    cols = st.columns(len(items))
    for col, m in zip(cols, items):
        with col:
            _html(f'<div class="card metric-card">'
                  f'<div class="metric-icon {m.get("tone", "indigo")}">{m["icon"]}</div>'
                  f'<div><div class="metric-value">{m["value"]}</div>'
                  f'<div class="metric-label">{m["label"]}</div>'
                  f'<div class="metric-hint">{m.get("hint", "")}</div></div></div>')


def warning_card(title: str, text: str, icon: str = "⚠️") -> None:
    """Bannière d'avertissement didactique mise en évidence."""
    _html(f'<div class="warning-card"><div class="w-icon">{icon}</div>'
          f'<div><div class="card-kicker">Contexte analytique fondamental</div>'
          f'<h3>{title}</h3><p>{text}</p></div></div>')


def badge(text: str, kind: str = "") -> str:
    """Retourne le HTML d'un badge stylisé (kind : champion | ok | warn | bad)."""
    return f'<span class="badge {kind}">{text}</span>'


def pipeline(steps: list[tuple[str, str]]) -> None:
    """Stepper visuel représentant les étapes du pipeline Data Mining."""
    inner = "".join(f'<div class="pipe-step"><b>{i + 1}</b><strong>{t}</strong><small>{d}</small></div>'
                    for i, (t, d) in enumerate(steps))
    _html(f'<div class="card"><div class="card-kicker">Pipeline de Modélisation & Validation</div>'
          f'<div class="pipeline">{inner}</div></div>')


def result_card(pred: int, score_text: str = "", proba: float | None = None,
                key_techs_found: list[str] | None = None) -> None:
    """Affiche la carte de résultat avec jauge dynamique et analyse de la règle déterministe."""
    is_positive = (pred == 1)
    kind = "positive" if is_positive else "negative"
    label = "✅ Candidat Employé (Classe 1)" if is_positive else "⚠️ Candidat Non Employé (Classe 0)"
    
    gauge = ""
    if proba is not None:
        gauge_class = f"gauge {kind}"
        gauge = (f'<div class="{gauge_class}" style="--score:{proba * 360:.0f}deg"><div>'
                 f'<strong>{proba:.0%}</strong><span>confiance</span></div></div>')
    
    key_box = ""
    if key_techs_found is not None:
        n_keys = len(key_techs_found)
        is_rule_met = n_keys >= 3
        count_cls = "match" if is_rule_met else "unmatch"
        rule_desc = "Règle déterministe déclenchée (≥ 3 clés)" if is_rule_met else "Seuil déterministe non atteint (< 3 clés)"
        techs_str = ", ".join(key_techs_found) if key_techs_found else "Aucune technologie clé"
        
        key_box = (f'<div class="key-tech-box">'
                   f'<div class="key-tech-header">'
                   f'<span>Technologies clés détectées</span>'
                   f'<span class="key-tech-count {count_cls}">{n_keys} / 10 clés</span>'
                   f'</div>'
                   f'<div style="font-size:0.78rem; color:var(--muted); line-height:1.4;">'
                   f'<strong>{rule_desc}</strong><br>'
                   f'<span>{techs_str}</span>'
                   f'</div></div>')

    _html(f'<div class="result-card"><div class="card-kicker">Diagnostic de l\'Inférence</div>'
          f'{gauge}<div class="result-badge {kind}">{label}</div>'
          f'<div class="result-note">{score_text}</div>'
          f'{key_box}'
          f'</div>')


def ethical_card(title: str, text: str, icon: str = "🛡️") -> None:
    """Bannière d'alerte éthique et IA responsable."""
    _html(f'<div class="ethical-card"><div class="w-icon">{icon}</div>'
          f'<div><div class="card-kicker">Avertissement Éthique & IA Responsable</div>'
          f'<h3>{title}</h3><p>{text}</p></div></div>')
