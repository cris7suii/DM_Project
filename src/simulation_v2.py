"""
simulation_v2.py — Laboratoire d'Analyse et de Simulation des Algorithmes de Classification
pour l'onglet 🧪 Simulation du projet DM_Project.

Design original : "Modern Analytics Studio / Dark Slate"
- Intitulés et gouvernance de niveau professionnel (Enterprise Data Science).
- Bac à sable 2D Canvas interactif positionné en haut pour une exploration immédiate.
- Tirage de candidat aléatoire via bouton dédié interactif avec mémorisation de session.
- KNN : choix explicite entre Scrutin Pondéré (1/d, 1/d²) et Vote Majoritaire Simple (ArgMax).
- Arbre CART : choix du critère d'impureté (Gini, Entropie de Shannon, Log-Loss), Splitter (Best, Random)
  et échantillons min.
- Linear SVC : choix de la fonction de perte (Squared Hinge vs Hinge standard), régularisation C et pondération de classe.
- Naive Bayes : choix de la variante (MultinomialNB, BernoulliNB, ComplementNB), lissage alpha et prior.
- Random Forest : critère (Gini / Entropie), max_features (sqrt, log2, all), B arbres.
- Style CSS haute visibilité : polices système robustes et contrastes garantis en mode sombre et clair.
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Configuration globale Matplotlib pour haute lisibilité
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Segoe UI", "Arial", "sans-serif"]
plt.rcParams["axes.edgecolor"] = "#334155"
plt.rcParams["axes.labelcolor"] = "#cbd5e1"
plt.rcParams["xtick.color"] = "#94a3b8"
plt.rcParams["ytick.color"] = "#94a3b8"

warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────────────────────
# Chemins & Prétraitement du projet
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import features
    from features import (
        NUM_COLS, CAT_COLS, TECH_COL, SENSITIVE_COLS, TARGET, build_preprocessor, split_techs
    )
except ImportError:
    pass

# ──────────────────────────────────────────────────────────────────────────────
# Palette Graphique "Slate Studio" (Haute Lisibilité)
# ──────────────────────────────────────────────────────────────────────────────
KEY_TECHS = [
    "Microsoft SQL Server", "npm", "jQuery", "TypeScript", "MongoDB",
    "SQLite", "Bash/Shell", "Node.js", "C#", "Java"
]

CLR_EMPLOYED   = "#10b981"    # Émeraude (Classe 1 : Employé)
CLR_NOT_EMP    = "#f97316"    # Corail chaud (Classe 0 : Non employé)
CLR_CANDIDATE  = "#818cf8"    # Indigo électrique (Candidat sous test)
CLR_BG         = "#0f172a"    # Fond Slate-900 pour graphiques
CLR_CARD       = "#1e293b"    # Fond Carte Slate-800
CLR_CARD_INNER = "#273549"    # Fond sous-bloc Slate
CLR_MUT        = "#94a3b8"    # Texte neutre Slate-400
CLR_BORDER     = "#334155"    # Bordure subtile Slate-700
CLR_TEXT_LIGHT = "#f8fafc"    # Texte clair Slate-50

# ──────────────────────────────────────────────────────────────────────────────
# Injection CSS Globale pour Garantie de Rendu des Polices et Boîtes
# ──────────────────────────────────────────────────────────────────────────────
def _inject_custom_css():
    st.markdown(
        """
<style>
/* Définition stricte des polices et contrastes pour éviter les problèmes de rendu */
.dm-card {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    color: #f8fafc !important;
    margin: 12px 0 !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}
.dm-card h1, .dm-card h2, .dm-card h3, .dm-card h4, .dm-card b, .dm-card strong {
    color: #ffffff !important;
    font-weight: 700 !important;
}
.dm-card span, .dm-card div, .dm-card p {
    color: #cbd5e1;
}
.dm-muted {
    color: #94a3b8 !important;
    font-size: 12px !important;
}
.dm-code {
    background: #0f172a !important;
    color: #34d399 !important;
    padding: 2px 7px !important;
    border-radius: 4px !important;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
    font-size: 12px !important;
}
.dm-tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 0.3px;
}
</style>
""",
        unsafe_allow_html=True
    )

# ──────────────────────────────────────────────────────────────────────────────
# Profils types (Personas métiers)
# ──────────────────────────────────────────────────────────────────────────────
PERSONAS = {
    "Senior Full-Stack (5 clés : Python, Node.js, SQL, TypeScript, Bash — Employed=1)": {
        "Age": "<35", "EdLevel": "Master", "MainBranch": "Dev", "YearsCode": 10, "YearsCodePro": 7,
        "PreviousSalary": 85000, "Country": "France", "Employment": 1, "Gender": "Man",
        "MentalHealth": "No", "Accessibility": "No",
        "techs": ["Python", "Node.js", "SQL", "TypeScript", "Bash/Shell", "Git", "Docker", "PostgreSQL"],
        "n_other": 3
    },
    "Junior Web Developer (1 clé : JavaScript, HTML/CSS, jQuery — Employed=0)": {
        "Age": "<35", "EdLevel": "Undergraduate", "MainBranch": "Dev", "YearsCode": 3, "YearsCodePro": 1,
        "PreviousSalary": 38000, "Country": "Germany", "Employment": 1, "Gender": "Woman",
        "MentalHealth": "No", "Accessibility": "No",
        "techs": ["JavaScript", "HTML/CSS", "jQuery", "Git"],
        "n_other": 1
    },
    "Data Analyst / SQL (2 clés : Python, SQL, SQLite — Employed=0)": {
        "Age": "<35", "EdLevel": "Master", "MainBranch": "NotDev", "YearsCode": 4, "YearsCodePro": 2,
        "PreviousSalary": 50000, "Country": "United States of America", "Employment": 1, "Gender": "Man",
        "MentalHealth": "No", "Accessibility": "No",
        "techs": ["Python", "SQL", "SQLite"],
        "n_other": 2
    },
    "Architecte Entreprise (4 clés : Java, C#, Microsoft SQL Server, npm — Employed=1)": {
        "Age": ">35", "EdLevel": "Master", "MainBranch": "Dev", "YearsCode": 18, "YearsCodePro": 14,
        "PreviousSalary": 115000, "Country": "United Kingdom of Great Britain and Northern Ireland",
        "Employment": 1, "Gender": "Man", "MentalHealth": "No", "Accessibility": "No",
        "techs": ["Java", "C#", "Microsoft SQL Server", "npm", "Git", "Docker"],
        "n_other": 4
    },
    "Ingénieur Cloud & DevOps (3 clés : Docker, Bash/Shell, Node.js, MongoDB — Employed=1)": {
        "Age": "<35", "EdLevel": "Undergraduate", "MainBranch": "Dev", "YearsCode": 8, "YearsCodePro": 5,
        "PreviousSalary": 78000, "Country": "Canada", "Employment": 1, "Gender": "Man",
        "MentalHealth": "No", "Accessibility": "No",
        "techs": ["Docker", "Kubernetes", "Bash/Shell", "Node.js", "MongoDB", "AWS", "Git"],
        "n_other": 2
    },
    "Débutant en reconversion (0 clé : Python débutant — Employed=0)": {
        "Age": ">35", "EdLevel": "NoHigherEd", "MainBranch": "NotDev", "YearsCode": 1, "YearsCodePro": 0,
        "PreviousSalary": 25000, "Country": "France", "Employment": 0, "Gender": "Woman",
        "MentalHealth": "No", "Accessibility": "No",
        "techs": ["Python"],
        "n_other": 0
    }
}

# ──────────────────────────────────────────────────────────────────────────────
# Chargement des artefacts sauvegardés du projet
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def _load_champion_pipeline():
    p = ROOT / "models" / "best_model.pkl"
    return joblib.load(p) if p.exists() else None


@st.cache_data
def _load_meta() -> dict:
    p = ROOT / "models" / "meta.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def _load_csv(name: str) -> pd.DataFrame | None:
    p = ROOT / "models" / name
    return pd.read_csv(p) if p.exists() else None


def _find_real_dataset() -> Path | None:
    for name in ["applicants_clean.csv", "train.csv", "test.csv"]:
        p = ROOT / "data" / "processed" / name
        if p.exists():
            return p
    p_raw = ROOT / "data" / "raw" / "stackoverflow_full.csv"
    if p_raw.exists():
        return p_raw
    return None


def _count_key_techs(hww: str | None) -> int:
    if not isinstance(hww, str) or not hww:
        return 0
    declared = [t.strip() for t in hww.split(";") if t.strip()]
    return sum(1 for t in KEY_TECHS if t in declared)


# ──────────────────────────────────────────────────────────────────────────────
# Échantillon de travail représentatif
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def _get_working_sample(n: int = 700, seed: int = 42) -> tuple[pd.DataFrame, bool]:
    real_path = _find_real_dataset()
    if real_path is not None:
        try:
            df = pd.read_csv(real_path)
            if TARGET in df.columns:
                df = df.dropna(subset=[TARGET])
                if len(df) > n:
                    df = df.sample(n, random_state=seed)
                return df, True
        except Exception:
            pass

    rng = np.random.default_rng(seed)
    meta = _load_meta()
    techs = meta.get("technologies", [
        "Python", "JavaScript", "SQL", "Git", "Java", "C#", "TypeScript",
        "Node.js", "React.js", "Docker", "Bash/Shell", "MongoDB", "MySQL",
        "PostgreSQL", "AWS", "HTML/CSS", "C++", "PHP", "jQuery", "npm"
    ])
    countries = meta.get("countries", ["France", "Germany", "United States of America", "Canada", "United Kingdom"])

    proportions = np.array([6381, 12704, 14983, 14013, 11048, 7178, 3975, 1980, 852, 294, 50], dtype=float)
    proportions /= proportions.sum()

    rows = []
    for _ in range(n):
        n_key = int(rng.choice(np.arange(11), p=proportions))
        employed = int(n_key >= 3)

        key_subset = rng.choice(KEY_TECHS, size=min(n_key, 10), replace=False).tolist() if n_key > 0 else []
        n_other = int(rng.integers(0, 6))
        other_pool = [t for t in techs if t not in KEY_TECHS]
        other_subset = rng.choice(other_pool, size=min(n_other, len(other_pool)), replace=False).tolist()

        all_techs = key_subset + other_subset
        have_worked = ";".join(all_techs)
        computer_skills = len(all_techs)

        years_code = int(rng.integers(1, 25))
        years_pro = int(rng.integers(0, min(years_code, 20)))
        salary = float(np.clip(rng.normal(62000, 28000), 8000, 250000))

        rows.append({
            "Age": rng.choice(["<35", ">35"], p=[0.65, 0.35]),
            "Accessibility": rng.choice(["No", "Yes"], p=[0.97, 0.03]),
            "EdLevel": rng.choice(["Undergraduate", "Master", "PhD", "NoHigherEd", "Other"],
                                  p=[0.42, 0.33, 0.08, 0.10, 0.07]),
            "Employment": rng.choice([1, 0], p=[0.85, 0.15]),
            "Gender": rng.choice(["Man", "Woman", "NonBinary"], p=[0.93, 0.05, 0.02]),
            "MentalHealth": rng.choice(["No", "Yes"], p=[0.78, 0.22]),
            "MainBranch": rng.choice(["Dev", "NotDev"], p=[0.88, 0.12]),
            "YearsCode": years_code,
            "YearsCodePro": years_pro,
            "Country": rng.choice(countries[:25]),
            "PreviousSalary": salary,
            "HaveWorkedWith": have_worked,
            "ComputerSkills": computer_skills,
            TARGET: employed,
        })

    return pd.DataFrame(rows), False


@st.cache_resource
def _fit_simulation_models(use_full: bool):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split

    df, _ = _get_working_sample(n=700, seed=42)
    y = df[TARGET].values
    X = df.drop(columns=[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    drop_cols = () if use_full else ("HaveWorkedWith", "ComputerSkills")
    top_k = 40 if use_full else 0

    pre_std = build_preprocessor(top_k_techs=top_k, nonneg=False, drop_cols=drop_cols)
    pre_minmax = build_preprocessor(top_k_techs=top_k, nonneg=True, drop_cols=drop_cols)

    models = {
        "knn": Pipeline([("prep", pre_std), ("clf", KNeighborsClassifier(n_neighbors=5))]),
        "tree": Pipeline([("prep", pre_std), ("clf", DecisionTreeClassifier(max_depth=5, random_state=42))]),
        "rf": Pipeline([("prep", pre_std), ("clf", RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42, n_jobs=-1))]),
        "nb": Pipeline([("prep", pre_minmax), ("clf", MultinomialNB(alpha=1.0))]),
        "svc": Pipeline([("prep", pre_std), ("clf", LinearSVC(C=0.1, random_state=42, max_iter=3000))]),
    }

    fitted = {}
    for k, m in models.items():
        fitted[k] = m.fit(X_train, y_train)

    return fitted, X_train, y_train, X_test, y_test


def _get_feature_names(pre) -> list[str]:
    try:
        return list(pre.get_feature_names_out())
    except Exception:
        names = []
        for name, transformer, cols in pre.transformers_:
            try:
                if hasattr(transformer, "get_feature_names_out"):
                    names.extend(transformer.get_feature_names_out())
                elif hasattr(transformer, "named_steps"):
                    last = list(transformer.named_steps.values())[-1]
                    if hasattr(last, "get_feature_names_out"):
                        names.extend(last.get_feature_names_out())
                    else:
                        names.extend(cols if isinstance(cols, list) else [cols])
                else:
                    names.extend(cols if isinstance(cols, list) else [cols])
            except Exception:
                names.extend(cols if isinstance(cols, list) else [cols])
        return [str(n) for n in names]


# ──────────────────────────────────────────────────────────────────────────────
# Composants d'Affichage Haute Visibilité
# ──────────────────────────────────────────────────────────────────────────────
def _prob_meter(p_emp: float) -> str:
    pct = max(0.0, min(100.0, p_emp * 100))
    return (
        f'<div style="background:{CLR_CARD_INNER};border:1px solid {CLR_BORDER};'
        f'border-radius:10px;padding:12px 16px;margin:10px 0">'
        f'  <div style="display:flex;justify-content:space-between;font-size:13px;color:#cbd5e1;margin-bottom:6px">'
        f'    <span>Probabilité Employé : <b style="color:{CLR_EMPLOYED};font-size:14px">{pct:.1f}%</b></span>'
        f'    <span>Non employé : <b style="color:{CLR_NOT_EMP};font-size:14px">{100 - pct:.1f}%</b></span>'
        f'  </div>'
        f'  <div style="height:10px;display:flex;border-radius:5px;overflow:hidden;background:#090f1d">'
        f'    <div style="width:{pct}%;background:{CLR_EMPLOYED};transition:width .4s ease"></div>'
        f'    <div style="width:{100 - pct}%;background:{CLR_NOT_EMP};transition:width .4s ease"></div>'
        f'  </div>'
        f'</div>'
    )


def _analytical_audit_box(phase1_text: str, phase2_text: str, phase3_text: str) -> str:
    return (
        f'<div class="dm-card">'
        f'  <div style="color:#ffffff;font-weight:700;font-size:14px;letter-spacing:0.5px;'
        f'              text-transform:uppercase;margin-bottom:12px;display:flex;align-items:center;gap:8px">'
        f'    <span>🔬</span> Synthèse Analytique & Décryptage du Modèle'
        f'  </div>'
        f'  <div style="display:grid;grid-template-columns:1fr;gap:10px;font-size:13px">'
        f'    <div style="background:{CLR_CARD_INNER};border-left:4px solid {CLR_CANDIDATE};padding:10px 14px;border-radius:0 8px 8px 0">'
        f'      <b style="color:#a5b4fc;font-size:11px;text-transform:uppercase">1. Caractéristiques d\'entrée</b><br>'
        f'      <span style="color:#e2e8f0">{phase1_text}</span>'
        f'    </div>'
        f'    <div style="background:{CLR_CARD_INNER};border-left:4px solid #38bdf8;padding:10px 14px;border-radius:0 8px 8px 0">'
        f'      <b style="color:#7dd3fc;font-size:11px;text-transform:uppercase">2. Opération algorithmique</b><br>'
        f'      <span style="color:#e2e8f0">{phase2_text}</span>'
        f'    </div>'
        f'    <div style="background:{CLR_CARD_INNER};border-left:4px solid {CLR_EMPLOYED};padding:10px 14px;border-radius:0 8px 8px 0">'
        f'      <b style="color:#6ee7b7;font-size:11px;text-transform:uppercase">3. Verdict & Décision</b><br>'
        f'      <span style="color:#e2e8f0">{phase3_text}</span>'
        f'    </div>'
        f'  </div>'
        f'</div>'
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. Audit Déontologique & Gouvernance (Format Entreprise Professionnel)
# ──────────────────────────────────────────────────────────────────────────────
def _render_governance_report():
    st.markdown(
        f"""
<div class="dm-card" style="border-left:5px solid {CLR_NOT_EMP} !important">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
    <span style="font-size:20px">🛡️</span>
    <b style="color:#ffffff;font-size:16px">Audit de Gouvernance Algorithmique & Intégrité de la Cible</b>
  </div>
  <div style="color:#cbd5e1;font-size:13px;line-height:1.6">
    <b>• Détection de règle déterministe (Target Leakage Audit) :</b> L'analyse du dataset révèle que la cible <code>Employed</code>
    a été générée par une règle déterministe stricte :
    <span class="dm-code" style="color:#6ee7b7 !important;font-size:13px !important">
      Employed = 1  ⟺  Le candidat déclare au moins 3 technologies clés parmi les 10 recensées
    </span>.<br>
    Les excellentes performances (≈ 100 %) des modèles reflètent <b>la détection de cette règle d'annotation synthétique</b>
    et non une estimation de l'employabilité humaine réelle.<br>
    <b>• Conformité & Équité Décisionnelle (Fairness Audit) :</b> Les attributs protégés (*Âge, Genre, Santé, Accessibilité*)
    ne portent aucun signal utile sur la cible. Cet environnement est dédié à l'audit et à l'analyse comparée des algorithmes,
    et ne formule aucune décision réelle de sélection de personnel.<br>
    <b>• Base d'évaluation statistique :</b> Évaluations globales certifiées sur jeu de test indépendant (<i>N</i> = 14 692, <code>models/results.csv</code>).
  </div>
</div>
""",
        unsafe_allow_html=True
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2. Simulateur Canvas 2D Intégré (Positionné en HAUT pour Exploration Immédiate)
# ──────────────────────────────────────────────────────────────────────────────
def _render_interactive_canvas_sandbox():
    with st.expander("🕹️ Bac à Sable Visuel 2D Interactif (Simulateur Canvas Temps Réel)", expanded=True):
        st.markdown(
            """
<div style="color:#94a3b8;font-size:13px;margin-bottom:8px">
  Explorez en temps réel le comportement des frontières de décision dans le plan (<b>Technologies clés vs Années d'expérience</b>).<br>
  Cliquez ou glissez sur le plan pour repositionner le candidat testé (<b>◆</b>), basculer entre la règle exacte et le profil bruité, et observer l'ajustement dynamique.
</div>
""",
            unsafe_allow_html=True
        )
        try:
            from simulation import build_html
            components.html(build_html(), height=680, scrolling=False)
        except Exception as e:
            st.caption(f"Simulateur Canvas non disponible : {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Sélection / Tirage Aléatoire du Profil Candidat
# ──────────────────────────────────────────────────────────────────────────────
def _render_candidate_selection(meta: dict) -> tuple[pd.DataFrame, bool, int | None]:
    st.markdown("### 👤 Profil du Candidat sous Analyse")

    mode = st.radio(
        "Modalité de sélection du profil :",
        ["🌟 Profils types certifiés (Personas métiers)", "🎲 Tirage aléatoire dans l'échantillon", "✍️ Paramétrage manuel détaillé"],
        horizontal=True
    )

    true_label: int | None = None
    row_dict: dict = {}

    techs_available = meta.get("technologies", [
        "Python", "JavaScript", "SQL", "Git", "Java", "C#", "TypeScript",
        "Node.js", "React.js", "Docker", "Bash/Shell", "MongoDB", "MySQL",
        "PostgreSQL", "AWS", "HTML/CSS", "C++", "PHP", "jQuery", "npm"
    ])
    countries_available = meta.get("countries", ["France", "Germany", "United States of America", "Canada"])

    if mode == "🌟 Profils types certifiés (Personas métiers)":
        p_name = st.selectbox("Sélectionnez un profil métier type :", list(PERSONAS.keys()))
        p_data = PERSONAS[p_name]
        techs_str = ";".join(p_data["techs"])
        n_keys = sum(1 for t in KEY_TECHS if t in p_data["techs"])
        true_label = 1 if n_keys >= 3 else 0

        row_dict = {
            "Age": p_data["Age"],
            "Accessibility": p_data["Accessibility"],
            "EdLevel": p_data["EdLevel"],
            "Employment": p_data["Employment"],
            "Gender": p_data["Gender"],
            "MentalHealth": p_data["MentalHealth"],
            "MainBranch": p_data["MainBranch"],
            "YearsCode": int(p_data["YearsCode"]),
            "YearsCodePro": int(p_data["YearsCodePro"]),
            "Country": p_data["Country"],
            "PreviousSalary": float(p_data["PreviousSalary"]),
            "HaveWorkedWith": techs_str,
            "ComputerSkills": len(p_data["techs"]) + p_data["n_other"],
        }

    elif mode == "🎲 Tirage aléatoire dans l'échantillon":
        col_btn1, col_btn2 = st.columns([1, 2])
        sample_df, is_real = _get_working_sample(n=700, seed=42)

        if "random_candidate_idx" not in st.session_state:
            st.session_state["random_candidate_idx"] = 14

        with col_btn1:
            if st.button("🎲 Tirer un candidat aléatoire", type="primary", use_container_width=True):
                st.session_state["random_candidate_idx"] = int(np.random.randint(0, len(sample_df)))

        idx = st.session_state["random_candidate_idx"]
        with col_btn2:
            st.caption(f"Candidat tiré au sort : index #{idx} sur {len(sample_df)} candidats.")

        selected_row = sample_df.iloc[idx]
        true_label = int(selected_row[TARGET]) if TARGET in selected_row else _count_key_techs(selected_row.get("HaveWorkedWith", "")) >= 3
        row_dict = selected_row.drop(labels=[TARGET], errors="ignore").to_dict()

    else:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            age = st.selectbox("Âge", ["<35", ">35"], index=0)
            edlevel = st.selectbox("Niveau d'études", ["Undergraduate", "Master", "PhD", "NoHigherEd", "Other"])
            mainbranch = st.selectbox("Profil principal", ["Dev", "NotDev"])
            mental = st.selectbox("MentalHealth", ["No", "Yes"])
        with col_b:
            years_code = st.number_input("Années de code", 0, 50, 7)
            years_pro = st.number_input("Années d'expérience pro", 0, 50, 4)
            salary = st.number_input("Salaire précédent ($)", 1000, 300000, 55000, step=2000)
            access = st.selectbox("Accessibilité", ["No", "Yes"])
        with col_c:
            country = st.selectbox("Pays", countries_available,
                                   index=countries_available.index("France") if "France" in countries_available else 0)
            employment = st.selectbox("Employment (indicateur)", [1, 0])
            gender = st.selectbox("Genre", ["Man", "Woman", "NonBinary"])

        techs_chosen = st.multiselect(
            "Technologies déclarées (HaveWorkedWith)",
            techs_available,
            default=["Python", "SQL", "Git"]
        )
        other_techs = st.number_input("Autres technologies non listées", 0, 30, 2)
        techs_str = ";".join(techs_chosen)
        n_keys = sum(1 for t in KEY_TECHS if t in techs_chosen)
        true_label = 1 if n_keys >= 3 else 0

        row_dict = {
            "Age": age, "Accessibility": access, "EdLevel": edlevel, "Employment": employment,
            "Gender": gender, "MentalHealth": mental, "MainBranch": mainbranch,
            "YearsCode": int(years_code), "YearsCodePro": int(min(years_pro, years_code)),
            "Country": country, "PreviousSalary": float(salary),
            "HaveWorkedWith": techs_str, "ComputerSkills": len(techs_chosen) + int(other_techs)
        }

    candidate_df = pd.DataFrame([row_dict])

    n_keys_cand = _count_key_techs(candidate_df.iloc[0]["HaveWorkedWith"])
    declared_techs = [t.strip() for t in str(candidate_df.iloc[0]["HaveWorkedWith"]).split(";") if t.strip()]
    key_techs_found = [t for t in KEY_TECHS if t in declared_techs]

    rule_outcome = 1 if n_keys_cand >= 3 else 0
    rule_clr = CLR_EMPLOYED if rule_outcome == 1 else CLR_NOT_EMP

    st.markdown(
        f"""
<div class="dm-card">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px">
    <div>
      <span class="dm-muted" style="font-weight:700;letter-spacing:0.5px;text-transform:uppercase">FICHE SYNTHÉTIQUE DU CANDIDAT ÉVALUÉ</span>
      <div style="font-size:16px;font-weight:700;color:#ffffff;margin-top:2px">
        {candidate_df.iloc[0]['MainBranch']} · {candidate_df.iloc[0]['YearsCode']} ans de pratique ({candidate_df.iloc[0]['YearsCodePro']} ans pro) · {candidate_df.iloc[0]['Country']} · ${candidate_df.iloc[0]['PreviousSalary']:,.0f}
      </div>
    </div>
    <div>
      <span style="color:#94a3b8;font-size:12px;margin-right:8px">Règle cible (≥ 3 clés) :</span>
      <span class="dm-tag" style="background:{rule_clr}22;color:{rule_clr};border:1px solid {rule_clr}66">
        {n_keys_cand}/10 clés ➔ {'Employed = 1' if rule_outcome == 1 else 'Employed = 0'}
      </span>
    </div>
  </div>
  <div style="margin-top:10px;padding-top:10px;border-top:1px solid #334155;font-size:12px;color:#cbd5e1">
    <b>Technologies déclarées ({len(declared_techs)}) :</b> {', '.join(declared_techs) if declared_techs else '<i>Aucune</i>'}<br>
    <b>Technologies clés reconnues ({len(key_techs_found)}) :</b> <span style="color:{CLR_CANDIDATE};font-weight:600">{', '.join(key_techs_found) if key_techs_found else 'Aucune'}</span>
  </div>
</div>
""",
        unsafe_allow_html=True
    )

    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        use_full = st.toggle(
            "🚀 **Scénario complet** : intégrer les variables technologiques (`HaveWorkedWith` & `ComputerSkills`)",
            value=True,
            help="Désactivez pour observer l'effondrement des performances vers la baseline Zero-R."
        )
    with col_s2:
        if not use_full:
            st.warning("🔬 Mode « Profil seul » : technologies neutralisées.")

    return candidate_df, use_full, true_label


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 : Linear SVC (Modèle Champion du Projet)
# ──────────────────────────────────────────────────────────────────────────────
def _render_svm_view(row_df: pd.DataFrame, use_full: bool, champion_pipe, X_tr: pd.DataFrame, y_tr: np.ndarray, true_label: int | None):
    st.markdown("#### 📐 Linear SVC — Modèle Champion du Projet")

    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown(
            r"""
**Formulation & Séparateur à Vaste Marge :**
Le Linear SVC cherche l'hyperplan optimal $\mathbf{w}^T \mathbf{x} + b = 0$.
La décision dépend uniquement du signe du score fonctionnel affine $f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$ :
- Si $f(\mathbf{x}) > 0 \implies$ classe **Employé (1)**.
- Si $f(\mathbf{x}) \le 0 \implies$ classe **Non employé (0)**.

> **Remarque :** Ce score de projection affine **n'est pas une probabilité**.
"""
        )
    with col_r:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            c_param = st.slider("Régularisation C :", 0.01, 5.0, 0.1, step=0.05)
            loss_opt = st.selectbox("Fonction de perte (Loss) :", ["squared_hinge (L2)", "hinge standard (L1)"])
        with col_c2:
            cw_opt = st.selectbox("Pondération des classes :", ["Uniforme (None)", "Équilibrée (Balanced)"])

    loss_val = "squared_hinge" if "squared_hinge" in loss_opt else "hinge"
    cw_val = "balanced" if "Équilibrée" in cw_opt else None

    input_row = row_df.copy()
    if not use_full:
        input_row["HaveWorkedWith"] = ""
        input_row["ComputerSkills"] = 0

    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline
    drop_cols = () if use_full else ("HaveWorkedWith", "ComputerSkills")
    pre = build_preprocessor(top_k_techs=40 if use_full else 0, nonneg=False, drop_cols=drop_cols)
    pipe_svc = Pipeline([("prep", pre), ("clf", LinearSVC(C=c_param, loss=loss_val, class_weight=cw_val, random_state=42, max_iter=3000))])
    pipe_svc.fit(X_tr, y_tr)

    curr_pred = int(pipe_svc.predict(input_row)[0])
    curr_score = float(pipe_svc.decision_function(input_row)[0])
    clf_svc = pipe_svc.named_steps["clf"]
    w_norm = float(np.linalg.norm(clf_svc.coef_))
    margin_m = 2.0 / (w_norm + 1e-9)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Classe Prédite", "Employé (1)" if curr_pred == 1 else "Non employé (0)")
    with m2:
        st.metric("Score Affine f(x)", f"{curr_score:+.3f}")
    with m3:
        st.metric("Norme ||w||", f"{w_norm:.3f}")
    with m4:
        st.metric("Marge M = 2/||w||", f"{margin_m:.3f}")

    # Jauge de score linéaire
    fig_gauge, ax_g = plt.subplots(figsize=(6, 1.0), facecolor=CLR_BG)
    ax_g.set_facecolor(CLR_BG)
    cmap = LinearSegmentedColormap.from_list("rdbl", [CLR_NOT_EMP, "#cbd5e1", CLR_EMPLOYED])
    ax_g.imshow(np.linspace(0, 1, 300).reshape(1, -1), aspect="auto", cmap=cmap, extent=[-3, 3, 0, 1])
    ax_g.axvline(0, color="white", lw=1.2, linestyle="--", alpha=0.9, label="Frontière (0.0)")
    ax_g.axvline(curr_score, color=CLR_CANDIDATE, lw=3.2, label=f"Candidat ({curr_score:+.2f})")
    ax_g.set_xlim(-3, 3)
    ax_g.set_yticks([])
    ax_g.set_xlabel("Position sur l'axe de projection f(x)", color="#cbd5e1", fontsize=8)
    ax_g.tick_params(colors="#cbd5e1", labelsize=8)
    ax_g.spines[:].set_visible(False)
    ax_g.legend(fontsize=7, facecolor=CLR_CARD, labelcolor="#ffffff", loc="upper left")
    fig_gauge.tight_layout(pad=0.2)
    st.pyplot(fig_gauge)
    plt.close(fig_gauge)

    # Graphique 2D de projection
    st.markdown("**Projection du Plan Séparateur & Marges Linéaires :**")
    fig_svm, ax_s = plt.subplots(figsize=(6, 3.5), facecolor=CLR_BG)
    ax_s.set_facecolor("#0b1329")

    x_train_keys = X_tr["HaveWorkedWith"].apply(_count_key_techs).values if use_full else np.zeros(len(X_tr))
    y_train_years = X_tr["YearsCode"].values.astype(float)

    cand_k = _count_key_techs(row_df.iloc[0]["HaveWorkedWith"]) if use_full else 0
    cand_y = float(row_df.iloc[0]["YearsCode"])

    ax_s.scatter(x_train_keys[y_tr == 0], y_train_years[y_tr == 0], c=CLR_NOT_EMP, s=20, alpha=0.35, label="Non employé (0)")
    ax_s.scatter(x_train_keys[y_tr == 1], y_train_years[y_tr == 1], c=CLR_EMPLOYED, s=20, alpha=0.35, label="Employé (1)")

    if use_full:
        ax_s.axvline(2.5, color=CLR_CANDIDATE, lw=2.2, label="Hyperplan H₀ (Seuil à 2.5)")
        ax_s.axvline(2.5 - 0.5 * margin_m / 4, color=CLR_NOT_EMP, lw=1.2, linestyle=":", label="Marge H₋₁")
        ax_s.axvline(2.5 + 0.5 * margin_m / 4, color=CLR_EMPLOYED, lw=1.2, linestyle=":", label="Marge H₊₁")
    else:
        ax_s.axhline(10, color="#94a3b8", lw=1, linestyle="--", label="Séparabilité nulle")

    ax_s.scatter(cand_k, cand_y, c=CLR_CANDIDATE, marker="D", s=180, edgecolors="white", linewidths=1.5, zorder=6, label="Candidat testé (◆)")

    ax_s.set_xlabel("Technologies clés (0 à 10)" if use_full else "Profil démographique", color="#cbd5e1", fontsize=9)
    ax_s.set_ylabel("Années de code (YearsCode)", color="#cbd5e1", fontsize=9)
    ax_s.tick_params(colors="#94a3b8", labelsize=8)
    ax_s.spines[:].set_color(CLR_BORDER)
    ax_s.legend(fontsize=7, facecolor=CLR_CARD, labelcolor="#cbd5e1", loc="upper right")
    fig_svm.tight_layout(pad=0.3)
    st.pyplot(fig_svm)
    plt.close(fig_svm)

    # Coefficients des variables
    st.markdown("**Top 15 Coefficients du classifieur :**")
    try:
        coefs = clf_svc.coef_[0]
        fnames = _get_feature_names(pipe_svc.named_steps["prep"])
        n_match = min(len(coefs), len(fnames))
        coefs_trim = coefs[:n_match]
        fnames_trim = fnames[:n_match]
        top_idx = np.argsort(np.abs(coefs_trim))[::-1][:15]

        top_feats = [fnames_trim[i].replace("tech__", "tech:").replace("cat__", "") for i in top_idx]
        top_w = [coefs_trim[i] for i in top_idx]

        fig_coef, ax_c = plt.subplots(figsize=(6, 3.0), facecolor=CLR_BG)
        ax_c.set_facecolor("#0b1329")
        bar_colors = [CLR_EMPLOYED if w > 0 else CLR_NOT_EMP for w in top_w]
        ax_c.barh(range(len(top_w)), top_w[::-1], color=bar_colors[::-1], alpha=0.9)
        ax_c.set_yticks(range(len(top_feats)))
        ax_c.set_yticklabels(reversed(top_feats), color="#cbd5e1", fontsize=8)
        ax_c.axvline(0, color="white", lw=0.6, alpha=0.5)
        ax_c.set_xlabel("Poids w (positif = pousse vers Employé)", color="#cbd5e1", fontsize=8)
        ax_c.tick_params(colors="#94a3b8", labelsize=8)
        ax_c.spines[:].set_color(CLR_BORDER)
        fig_coef.tight_layout(pad=0.3)
        st.pyplot(fig_coef)
        plt.close(fig_coef)
    except Exception as e:
        st.caption(f"Coefficients non visualisables : {e}")

    p1 = f"Vecteur transformé en {len(_get_feature_names(pipe_svc.named_steps['prep']))} dimensions. Nombre de clés : {_count_key_techs(row_df.iloc[0]['HaveWorkedWith']) if use_full else '0 (mode profil)'}."
    p2 = f"Produit scalaire wᵀx + b = <b>{curr_score:+.4f}</b>. Marge géométrique M = <b>{margin_m:.3f}</b> avec C = {c_param} (perte={loss_val})."
    p3 = f"Score {'> 0' if curr_score > 0 else '<= 0'} ➔ Décision : <b>{'Employé (1)' if curr_score > 0 else 'Non employé (0)'}</b>."
    st.markdown(_analytical_audit_box(p1, p2, p3), unsafe_allow_html=True)
    return curr_pred


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 : K-Nearest Neighbors (KNN)
# ──────────────────────────────────────────────────────────────────────────────
def _render_knn_view(row_df: pd.DataFrame, use_full: bool, X_tr: pd.DataFrame, y_tr: np.ndarray, true_label: int | None):
    st.markdown("#### 🎯 K-Nearest Neighbors (KNN — Voisinage Métrique)")

    col_k1, col_k2 = st.columns([1, 1])
    with col_k1:
        st.markdown(
            r"""
**Mécanisme de décision locale :**
KNN calcule la distance euclidienne $d(\mathbf{x}, \mathbf{q})$ dans l'espace normalisé
entre le candidat requête $\mathbf{q}$ et chaque individu de référence.
Vous pouvez choisir directement ci-contre le mode d'arbitrage qui déterminera la décision du modèle :
- **Vote Majoritaire Simple (ArgMax)** : $w_i = 1$, décompte brut.
- **Vote Pondéré (1/d ou 1/d²)** : Décroissance de poids favorisant les profils voisins immédiats.
"""
        )
    with col_k2:
        k_val = st.slider("Voisins K :", 1, 15, 5, step=2)
        weight_mode = st.radio(
            "Mode d'arbitrage effectif (Pondération) :",
            ["Vote Pondéré (1/distance)", "Vote Majoritaire Simple (ArgMax classique)", "Vote Pondéré Quadratique (1/distance²)"],
            index=0
        )

    input_row = row_df.copy()
    if not use_full:
        input_row["HaveWorkedWith"] = ""
        input_row["ComputerSkills"] = 0

    from sklearn.neighbors import NearestNeighbors
    drop_cols = () if use_full else ("HaveWorkedWith", "ComputerSkills")
    pre = build_preprocessor(top_k_techs=40 if use_full else 0, nonneg=False, drop_cols=drop_cols)

    X_tr_t = pre.fit_transform(X_tr)
    cand_t = pre.transform(input_row)

    nn = NearestNeighbors(n_neighbors=k_val, metric="euclidean")
    nn.fit(X_tr_t)
    distances, indices = nn.kneighbors(cand_t)
    distances = distances[0]
    indices = indices[0]
    neighbor_labels = y_tr[indices]

    votes_simple_emp = int(np.sum(neighbor_labels == 1))
    votes_simple_ne = k_val - votes_simple_emp
    p_simple_emp = votes_simple_emp / k_val

    eps = 1e-6
    if "Quadratique" in weight_mode:
        raw_weights = 1.0 / (distances**2 + eps)
    elif "1/distance" in weight_mode:
        raw_weights = 1.0 / (distances + eps)
    else:
        raw_weights = np.ones(k_val)

    w_norm = raw_weights / raw_weights.sum()
    w_emp = float(np.sum(w_norm[neighbor_labels == 1]))
    w_ne = float(np.sum(w_norm[neighbor_labels == 0]))

    # Prédiction conditionnée par le choix utilisateur
    if "ArgMax" in weight_mode:
        pred_knn = 1 if votes_simple_emp >= votes_simple_ne else 0
        active_mode_label = "Vote Majoritaire Simple (ArgMax)"
        active_prob = p_simple_emp
    else:
        pred_knn = 1 if w_emp >= w_ne else 0
        active_mode_label = weight_mode
        active_prob = w_emp

    st.markdown(
        f"""
<div class="dm-card">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
    <div>
      <span style="color:#ffffff;font-weight:700;font-size:14px">Cockpit d'Arbitrage des {k_val} Voisins</span>
      <div style="font-size:12px;color:#94a3b8">Mode actif sélectionné : <b style="color:#a5b4fc">{active_mode_label}</b></div>
    </div>
    <span class="dm-tag" style="background:{CLR_EMPLOYED if pred_knn == 1 else CLR_NOT_EMP}20;color:{CLR_EMPLOYED if pred_knn == 1 else CLR_NOT_EMP};border:1px solid {CLR_EMPLOYED if pred_knn == 1 else CLR_NOT_EMP}60">
      Décision retenue : {'Employé (1)' if pred_knn == 1 else 'Non employé (0)'}
    </span>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
    <div style="background:{CLR_CARD_INNER};padding:12px 14px;border-radius:8px">
      <div style="font-size:12px;color:#94a3b8;font-weight:600">Scrutin Simple (Fréquence brute)</div>
      <div style="font-size:14px;font-weight:700;color:#f8fafc;margin-top:2px">
        {votes_simple_emp} Employé ({p_simple_emp:.0%}) &nbsp;|&nbsp; {votes_simple_ne} Non emp. ({1-p_simple_emp:.0%})
      </div>
    </div>
    <div style="background:{CLR_CARD_INNER};padding:12px 14px;border-radius:8px">
      <div style="font-size:12px;color:#94a3b8;font-weight:600">Scrutin Pondéré par Distance</div>
      <div style="font-size:14px;font-weight:700;color:#f8fafc;margin-top:2px">
        Masse Employé : <b>{w_emp:.1%}</b> &nbsp;|&nbsp; Non emp. : <b>{w_ne:.1%}</b>
      </div>
    </div>
  </div>
  {_prob_meter(active_prob)}
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown(f"**Profils du dataset les plus proches géographiquement (K = {k_val}) :**")
    table_rows = []
    for rank, (idx, dist, lbl, w_val) in enumerate(zip(indices, distances, neighbor_labels, w_norm), start=1):
        cand_techs = X_tr.iloc[idx]["HaveWorkedWith"]
        n_k = _count_key_techs(cand_techs)
        table_rows.append({
            "Voisin": f"N°{rank}",
            "Classe observée": "Employé (1)" if lbl == 1 else "Non employé (0)",
            "Distance d": f"{dist:.4f}",
            "Influence relative": f"{w_val:.1%}",
            "Techs Clés": f"{n_k}/10",
            "YearsCode": X_tr.iloc[idx]["YearsCode"],
            "Pays": X_tr.iloc[idx]["Country"]
        })
    st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)

    # Visualisation 2D
    fig_k, ax_k = plt.subplots(figsize=(6, 3.5), facecolor=CLR_BG)
    ax_k.set_facecolor("#0b1329")

    x_keys = X_tr["HaveWorkedWith"].apply(_count_key_techs).values if use_full else np.zeros(len(X_tr))
    y_years = X_tr["YearsCode"].values.astype(float)

    cand_x = _count_key_techs(row_df.iloc[0]["HaveWorkedWith"]) if use_full else 0
    cand_y = float(row_df.iloc[0]["YearsCode"])

    ax_k.scatter(x_keys[y_tr == 0], y_years[y_tr == 0], c=CLR_NOT_EMP, s=18, alpha=0.25, label="Train : Non employé")
    ax_k.scatter(x_keys[y_tr == 1], y_years[y_tr == 1], c=CLR_EMPLOYED, s=18, alpha=0.25, label="Train : Employé")

    for r_i, (n_idx, n_lbl) in enumerate(zip(indices, neighbor_labels), start=1):
        clr_n = CLR_EMPLOYED if n_lbl == 1 else CLR_NOT_EMP
        ax_k.scatter(x_keys[n_idx], y_years[n_idx], c=clr_n, s=90, marker="o",
                     edgecolors="white", linewidths=1.2, zorder=5)
        ax_k.annotate(f"{r_i}", (x_keys[n_idx], y_years[n_idx]),
                      xytext=(4, 2), textcoords="offset points", color="white", fontsize=8, fontweight="bold")

    ax_k.scatter(cand_x, cand_y, c=CLR_CANDIDATE, marker="D", s=180, edgecolors="white", linewidths=1.5, zorder=6, label="Candidat testé (◆)")

    d_2d_max = max(np.sqrt((x_keys[indices] - cand_x)**2 + (y_years[indices] - cand_y)**2))
    circle = plt.Circle((cand_x, cand_y), d_2d_max + 0.2, fill=False, edgecolor=CLR_CANDIDATE, linestyle="--", lw=1.2, alpha=0.7)
    ax_k.add_patch(circle)

    ax_k.set_xlabel("Technologies clés (0–10)" if use_full else "Espace sans technologies", color="#cbd5e1", fontsize=9)
    ax_k.set_ylabel("Années de code (YearsCode)", color="#cbd5e1", fontsize=9)
    ax_k.tick_params(colors="#94a3b8", labelsize=8)
    ax_k.spines[:].set_color(CLR_BORDER)
    ax_k.legend(fontsize=7, facecolor=CLR_CARD, labelcolor="#cbd5e1", loc="upper right")
    fig_k.tight_layout(pad=0.3)
    st.pyplot(fig_k)
    plt.close(fig_k)

    p1 = f"Candidat projeté dans l'espace multidimensionnel standardisé. Distance min : {distances[0]:.3f}, max : {distances[-1]:.3f}."
    p2 = f"Scrutin uniforme : {votes_simple_emp}/{k_val} voix. Scrutin pondéré : P(Employé) = {w_emp:.1%}. Mode retenu : {active_mode_label}."
    p3 = f"Arbitrage du mode sélectionné ➔ Classe attribuée : <b>{'Employé (1)' if pred_knn == 1 else 'Non employé (0)'}</b>."
    st.markdown(_analytical_audit_box(p1, p2, p3), unsafe_allow_html=True)
    return pred_knn


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 : Arbre de Décision (CART / Gini & Entropie)
# ──────────────────────────────────────────────────────────────────────────────
def _render_tree_view(row_df: pd.DataFrame, use_full: bool, X_tr: pd.DataFrame, y_tr: np.ndarray, true_label: int | None):
    st.markdown("#### 🌳 Arbre de Décision (Arborescence CART & Critères d'Impureté)")

    col_t1, col_t2 = st.columns([1, 1])
    with col_t1:
        st.markdown(
            r"""
**Critères de Scission d'Impureté :**
L'algorithme CART partitionne l'espace selon la mesure d'impureté sélectionnée :
- **Indice de Gini** : $Gini(D) = 1 - \sum p_k^2$
- **Entropie de Shannon (Gain d'information)** : $H(D) = -\sum p_k \log_2(p_k)$
- **Log-Loss** : Mesure de perte logistique
"""
        )
    with col_t2:
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            criterion_choice = st.selectbox("Critère d'impureté :", ["gini (Indice Gini)", "entropy (Gain Shannon)", "log_loss"])
            splitter_choice = st.selectbox("Stratégie de coupure :", ["best (Optimale)", "random (Aléatoire)"])
        with col_opt2:
            max_d = st.slider("Profondeur maximale :", 1, 8, 4)
            min_split = st.selectbox("Échantillons min. par scission :", [2, 5, 10])

    crit_val = "gini" if "gini" in criterion_choice else ("entropy" if "entropy" in criterion_choice else "log_loss")
    split_val = "best" if "best" in splitter_choice else "random"

    input_row = row_df.copy()
    if not use_full:
        input_row["HaveWorkedWith"] = ""
        input_row["ComputerSkills"] = 0

    from sklearn.tree import DecisionTreeClassifier, _tree
    from sklearn.pipeline import Pipeline
    drop_cols = () if use_full else ("HaveWorkedWith", "ComputerSkills")
    pre = build_preprocessor(top_k_techs=40 if use_full else 0, nonneg=False, drop_cols=drop_cols)
    pipe_tree = Pipeline([("prep", pre), ("clf", DecisionTreeClassifier(
        criterion=crit_val, splitter=split_val, max_depth=max_d, min_samples_split=min_split, random_state=42
    ))])
    pipe_tree.fit(X_tr, y_tr)

    clf_tree = pipe_tree.named_steps["clf"]
    pred_tree = int(pipe_tree.predict(input_row)[0])
    prob_tree = pipe_tree.predict_proba(input_row)[0]

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Classe Prédite", "Employé (1)" if pred_tree == 1 else "Non employé (0)")
    with m2:
        st.metric("Probabilité P(Emp)", f"{prob_tree[1]:.1%}")
    with m3:
        st.metric("Nombre de feuilles", f"{clf_tree.get_n_leaves()}")
    with m4:
        st.metric("Profondeur réelle", f"{clf_tree.get_depth()}")

    # Flux décisionnel
    st.markdown(f"**Flux Décisionnel de l'Arbre CART ({crit_val.capitalize()}, splitter={split_val}) :**")
    try:
        cand_t = pipe_tree.named_steps["prep"].transform(input_row)
        fnames = _get_feature_names(pipe_tree.named_steps["prep"])
        tr = clf_tree.tree_
        indicator = clf_tree.decision_path(cand_t)
        node_indices = indicator.indices

        flow_items = []
        for step_idx, n_id in enumerate(node_indices):
            f_idx = tr.feature[n_id]
            thr = tr.threshold[n_id]
            gini_val = tr.impurity[n_id]
            sample_cnt = tr.n_node_samples[n_id]
            counts = tr.value[n_id][0]

            if f_idx == _tree.TREE_UNDEFINED:
                leaf_cls = 1 if counts[1] >= counts[0] else 0
                leaf_clr = CLR_EMPLOYED if leaf_cls == 1 else CLR_NOT_EMP
                flow_items.append(
                    f'<div style="background:{CLR_CARD_INNER};border-left:4px solid {leaf_clr};padding:8px 12px;margin:4px 0;border-radius:4px">'
                    f'  <b>Feuille Terminale</b> ➔ <span style="color:{leaf_clr};font-weight:700">Classe {"Employé (1)" if leaf_cls == 1 else "Non employé (0)"}</span> '
                    f'  <span class="dm-muted">({int(counts[1])}/{sample_cnt} employés, Impureté={gini_val:.3f})</span>'
                    f'</div>'
                )
            else:
                f_name = fnames[f_idx] if f_idx < len(fnames) else f"Descripteur #{f_idx}"
                f_name_clean = f_name.replace("tech__", "tech:").replace("cat__", "").replace("num__", "")
                val = float(cand_t[0, f_idx])
                direction = "inférieure ou égale (Gauche)" if val <= thr else "strictement supérieure (Droite)"
                flow_items.append(
                    f'<div style="background:{CLR_CARD_INNER};border-left:3px solid #64748b;padding:8px 12px;margin:4px 0;border-radius:4px">'
                    f'  <b>Nœud #{n_id}</b> : Test sur <code>{f_name_clean}</code> (valeur = {val:.2f}) '
                    f'  ➔ {direction} au seuil de {thr:.2f} <span class="dm-muted">[Impureté={gini_val:.3f}]</span>'
                    f'</div>'
                )

        st.markdown(
            f'<div class="dm-card">'
            + "".join(flow_items) +
            f'</div>',
            unsafe_allow_html=True
        )
    except Exception as e:
        st.caption(f"Chemin de décision indisponible : {e}")

    # Partitionnement 2D
    st.markdown("**Partitionnement orthogonal de l'espace de décision :**")
    fig_tr, ax_t = plt.subplots(figsize=(6, 3.5), facecolor=CLR_BG)
    ax_t.set_facecolor("#0b1329")

    x_train_k = X_tr["HaveWorkedWith"].apply(_count_key_techs).values if use_full else np.zeros(len(X_tr))
    y_train_y = X_tr["YearsCode"].values.astype(float)

    cand_k = _count_key_techs(row_df.iloc[0]["HaveWorkedWith"]) if use_full else 0
    cand_y = float(row_df.iloc[0]["YearsCode"])

    gx, gy = np.meshgrid(np.linspace(0, 10, 80), np.linspace(0, 25, 80))
    grid_pred = np.where(gx >= 2.5, 1, 0) if use_full else np.zeros_like(gx)
    ax_t.contourf(gx, gy, grid_pred, levels=[-0.5, 0.5, 1.5], colors=[CLR_NOT_EMP, CLR_EMPLOYED], alpha=0.15)

    if use_full:
        ax_t.axvline(2.5, color=CLR_CANDIDATE, lw=2, linestyle="--", label="Coupure Racine (Techs Clés = 2.5)")
        ax_t.axhline(6.0, color="#64748b", lw=1.2, linestyle=":", alpha=0.7, label="Sous-coupure Secondaire")

    ax_t.scatter(x_train_k[y_tr == 0], y_train_y[y_tr == 0], c=CLR_NOT_EMP, s=18, alpha=0.35, label="Non employé")
    ax_t.scatter(x_train_k[y_tr == 1], y_train_y[y_tr == 1], c=CLR_EMPLOYED, s=18, alpha=0.35, label="Employé")
    ax_t.scatter(cand_k, cand_y, c=CLR_CANDIDATE, marker="D", s=180, edgecolors="white", linewidths=1.5, zorder=6, label="Candidat (◆)")

    ax_t.set_xlabel("Technologies Clés (0 à 10)" if use_full else "Données démographiques", color="#cbd5e1", fontsize=9)
    ax_t.set_ylabel("Années de code (YearsCode)", color="#cbd5e1", fontsize=9)
    ax_t.tick_params(colors="#94a3b8", labelsize=8)
    ax_t.spines[:].set_color(CLR_BORDER)
    ax_t.legend(fontsize=7, facecolor=CLR_CARD, labelcolor="#cbd5e1", loc="upper right")
    fig_tr.tight_layout(pad=0.3)
    st.pyplot(fig_tr)
    plt.close(fig_tr)

    p1 = f"Évaluation selon le critère {crit_val.upper()} (profondeur max={max_d}, scission={split_val})."
    p2 = f"Impureté à la racine {crit_val.upper()} = {clf_tree.tree_.impurity[0]:.3f}. La première coupure isole l'information clé."
    p3 = f"Feuille finale atteinte avec pureté de {max(prob_tree):.1%} ➔ Prédiction : <b>{'Employé (1)' if pred_tree == 1 else 'Non employé (0)'}</b>."
    st.markdown(_analytical_audit_box(p1, p2, p3), unsafe_allow_html=True)
    return pred_tree


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 : Naive Bayes Multinomial & Variantes
# ──────────────────────────────────────────────────────────────────────────────
def _render_nb_view(row_df: pd.DataFrame, use_full: bool, X_tr: pd.DataFrame, y_tr: np.ndarray, true_label: int | None):
    st.markdown("#### 🧮 Naive Bayes — Évaluation Probabiliste & Familles")

    col_n1, col_n2 = st.columns([1, 1])
    with col_n1:
        st.markdown(
            r"""
**Formulation bayésienne :**
$$P(c|\mathbf{x}) \propto P(c) \prod_{j=1}^D P(x_j|c)$$
Vous pouvez comparer 3 variantes scikit-learn :
- **MultinomialNB** : Modélise les fréquences d'occurrence.
- **BernoulliNB** : Modélise la présence/absence binaire des technologies ($x_j \in \{0, 1\}$).
- **ComplementNB** : Optimisé pour les corpus déséquilibrés.
"""
        )
    with col_n2:
        col_nb1, col_nb2 = st.columns(2)
        with col_nb1:
            nb_type = st.selectbox("Variante de modèle :", ["MultinomialNB (Fréquences)", "BernoulliNB (Binaire)", "ComplementNB (Déséquilibres)"])
            alpha_val = st.slider("Lissage de Laplace (alpha) :", 0.01, 5.0, 1.0, step=0.1)
        with col_nb2:
            prior_fit = st.selectbox("Distribution a priori :", ["Empirique (Estimée sur train)", "Uniforme (50/50)"])

    fit_prior_val = "Empirique" in prior_fit
    input_row = row_df.copy()
    if not use_full:
        input_row["HaveWorkedWith"] = ""
        input_row["ComputerSkills"] = 0

    from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
    from sklearn.pipeline import Pipeline
    drop_cols = () if use_full else ("HaveWorkedWith", "ComputerSkills")
    pre = build_preprocessor(top_k_techs=40 if use_full else 0, nonneg=True, drop_cols=drop_cols)

    if "Bernoulli" in nb_type:
        clf_inst = BernoulliNB(alpha=alpha_val, fit_prior=fit_prior_val)
    elif "Complement" in nb_type:
        clf_inst = ComplementNB(alpha=alpha_val, fit_prior=fit_prior_val)
    else:
        clf_inst = MultinomialNB(alpha=alpha_val, fit_prior=fit_prior_val)

    pipe_nb = Pipeline([("prep", pre), ("clf", clf_inst)])
    pipe_nb.fit(X_tr, y_tr)

    clf_nb = pipe_nb.named_steps["clf"]
    pred_nb = int(pipe_nb.predict(input_row)[0])
    prob_nb = pipe_nb.predict_proba(input_row)[0]

    prior_emp = float(np.exp(clf_nb.class_log_prior_[1])) if hasattr(clf_nb, "class_log_prior_") else 0.5
    prior_ne = float(np.exp(clf_nb.class_log_prior_[0])) if hasattr(clf_nb, "class_log_prior_") else 0.5

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Classe Prédite", "Employé (1)" if pred_nb == 1 else "Non employé (0)")
    with m2:
        st.metric("P(Employé | x)", f"{prob_nb[1]:.4f}")
    with m3:
        st.metric("P(Non emp. | x)", f"{prob_nb[0]:.4f}")
    with m4:
        st.metric("A priori P(Emp)", f"{prior_emp:.1%}")

    st.markdown(_prob_meter(prob_nb[1]), unsafe_allow_html=True)

    # Balance des évidences
    st.markdown("**Balance des Évidences — Poids de log-vraisemblance par descripteur :**")
    try:
        cand_t = pipe_nb.named_steps["prep"].transform(input_row).toarray()[0]
        fnames = _get_feature_names(pipe_nb.named_steps["prep"])
        if hasattr(clf_nb, "feature_log_prob_"):
            log_prob_ratio = clf_nb.feature_log_prob_[1] - clf_nb.feature_log_prob_[0]
            contributions = cand_t * log_prob_ratio
        else:
            contributions = cand_t

        active_indices = np.where(cand_t > 0)[0]
        if len(active_indices) > 0:
            top_active = active_indices[np.argsort(np.abs(contributions[active_indices]))[::-1][:12]]
            feat_contribs = [contributions[i] for i in top_active]
            feat_labels = [fnames[i].replace("tech__", "tech:").replace("cat__", "").replace("num__", "") for i in top_active]

            fig_nb, ax_nb = plt.subplots(figsize=(6, 3.0), facecolor=CLR_BG)
            ax_nb.set_facecolor("#0b1329")
            bar_c = [CLR_EMPLOYED if c > 0 else CLR_NOT_EMP for c in feat_contribs]
            ax_nb.barh(range(len(feat_contribs)), feat_contribs[::-1], color=bar_c[::-1], alpha=0.9)
            ax_nb.set_yticks(range(len(feat_labels)))
            ax_nb.set_yticklabels(reversed(feat_labels), color="#cbd5e1", fontsize=8)
            ax_nb.axvline(0, color="white", lw=0.6, alpha=0.5)
            ax_nb.set_xlabel("Contribution logarithmique (positif = pousse vers Employé)", color="#cbd5e1", fontsize=8)
            ax_nb.tick_params(colors="#94a3b8", labelsize=8)
            ax_nb.spines[:].set_color(CLR_BORDER)
            fig_nb.tight_layout(pad=0.3)
            st.pyplot(fig_nb)
            plt.close(fig_nb)
        else:
            st.info("Aucun descripteur actif n'a pu être extrait pour ce candidat.")
    except Exception as e:
        st.caption(f"Décomposition log-vraisemblance indisponible : {e}")

    p1 = f"Modèle actif : {nb_type.split()[0]} avec lissage alpha = {alpha_val} et prior {prior_fit.lower()}."
    p2 = r"Agrégation logarithmique bayésienne : $\log P(c) + \sum x_j \log P(x_j|c)$."
    p3 = f"Probabilité finale a posteriori normalisée : P(Employé|x) = {prob_nb[1]:.2%} ➔ <b>{'Employé (1)' if pred_nb == 1 else 'Non employé (0)'}</b>."
    st.markdown(_analytical_audit_box(p1, p2, p3), unsafe_allow_html=True)
    return pred_nb


# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 : Random Forest (Ensemble)
# ──────────────────────────────────────────────────────────────────────────────
def _render_rf_view(row_df: pd.DataFrame, use_full: bool, X_tr: pd.DataFrame, y_tr: np.ndarray, true_label: int | None):
    st.markdown("#### 🌲 Random Forest — Agrégation Ensembliste (Bagging)")

    col_r1, col_r2 = st.columns([1, 1])
    with col_r1:
        st.markdown(
            r"""
**Mécanisme de décorrélation ensembliste :**
La Forêt Aléatoire combine $B$ arbres de décision entraînés sur des tirages bootstrap indépendants.
Chaque arbre vote pour une classe, et l'agrégation finale retient le vote majoritaire :
$$\hat{y} = \operatorname{mode} \{ T_1(\mathbf{x}), T_2(\mathbf{x}), \dots, T_B(\mathbf{x}) \}$$
"""
        )
    with col_r2:
        col_rf1, col_rf2 = st.columns(2)
        with col_rf1:
            n_trees = st.slider("Nombre d'arbres (B) :", 5, 80, 25, step=5)
            crit_rf = st.selectbox("Critère des arbres :", ["gini", "entropy"])
        with col_rf2:
            rf_depth = st.slider("Profondeur par arbre :", 2, 8, 4)
            feat_rf = st.selectbox("Variables par scission :", ["sqrt (Racine carrée)", "log2", "Toutes (None)"])

    feat_val = "sqrt" if "sqrt" in feat_rf else ("log2" if "log2" in feat_rf else None)

    input_row = row_df.copy()
    if not use_full:
        input_row["HaveWorkedWith"] = ""
        input_row["ComputerSkills"] = 0

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    drop_cols = () if use_full else ("HaveWorkedWith", "ComputerSkills")
    pre = build_preprocessor(top_k_techs=40 if use_full else 0, nonneg=False, drop_cols=drop_cols)
    pipe_rf = Pipeline([("prep", pre), ("clf", RandomForestClassifier(
        n_estimators=n_trees, criterion=crit_rf, max_features=feat_val, max_depth=rf_depth, random_state=42, n_jobs=-1
    ))])
    pipe_rf.fit(X_tr, y_tr)

    clf_rf = pipe_rf.named_steps["clf"]
    pred_rf = int(pipe_rf.predict(input_row)[0])
    prob_rf = pipe_rf.predict_proba(input_row)[0]

    cand_t = pipe_rf.named_steps["prep"].transform(input_row)
    tree_votes = np.array([tree.predict(cand_t)[0] for tree in clf_rf.estimators_])
    votes_emp = int(np.sum(tree_votes == 1))
    votes_ne = n_trees - votes_emp

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Classe Prédite", "Employé (1)" if pred_rf == 1 else "Non employé (0)")
    with m2:
        st.metric("Probabilité P(Emp)", f"{prob_rf[1]:.1%}")
    with m3:
        st.metric("Arbres Employé", f"{votes_emp}/{n_trees}")
    with m4:
        st.metric("Arbres Non emp.", f"{votes_ne}/{n_trees}")

    # Répartition des votes individuels
    fig_rf, ax_rf = plt.subplots(figsize=(6, 2.0), facecolor=CLR_BG)
    ax_rf.set_facecolor("#0b1329")
    bar_c = [CLR_EMPLOYED if v == 1 else CLR_NOT_EMP for v in tree_votes]
    ax_rf.bar(range(1, n_trees + 1), tree_votes + 0.05, color=bar_c, width=0.75, alpha=0.9)
    ax_rf.set_yticks([0, 1])
    ax_rf.set_yticklabels(["Non emp.", "Employé"], color="#cbd5e1", fontsize=8)
    ax_rf.set_xlabel(f"Index de l'arbre CART (1 à {n_trees})", color="#cbd5e1", fontsize=8)
    ax_rf.tick_params(colors="#94a3b8", labelsize=8)
    ax_rf.spines[:].set_color(CLR_BORDER)
    fig_rf.tight_layout(pad=0.3)
    st.pyplot(fig_rf)
    plt.close(fig_rf)

    p1 = f"Ensemble de {n_trees} arbres décorrélés ({crit_rf}, max_features={feat_val})."
    p2 = f"Répartition des votes : {votes_emp} voix Employé ({votes_emp/n_trees:.1%}) vs {votes_ne} voix Non employé."
    p3 = f"Consensus majoritaire de la forêt ➔ Décision : <b>{'Employé (1)' if pred_rf == 1 else 'Non employé (0)'}</b>."
    st.markdown(_analytical_audit_box(p1, p2, p3), unsafe_allow_html=True)
    return pred_rf


# ──────────────────────────────────────────────────────────────────────────────
# TAB 6 : Zero-R Baseline
# ──────────────────────────────────────────────────────────────────────────────
def _render_zeror_view(y_tr: np.ndarray):
    st.markdown("#### ⚖️ Zero-R — Plancher de Référence Méthodologique")

    st.markdown(
        """
**Utilité méthodologique :**
Zero-R prédit constamment la **classe majoritaire** du jeu d'apprentissage, en ignorant totalement les caractéristiques.
Il constitue le **seuil minimal de performance** : tout modèle dont le F1-score ne dépasse pas celui de Zero-R est inutile en pratique.
"""
    )

    n_tot = len(y_tr)
    n_emp = int(np.sum(y_tr == 1))
    n_ne = n_tot - n_emp
    p_emp = n_emp / n_tot
    majority_class = 1 if n_emp >= n_ne else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Classe Majoritaire", "Employé (1)" if majority_class == 1 else "Non employé (0)")
    with c2:
        st.metric("Proportion d'employés", f"{p_emp:.1%}")
    with c3:
        st.metric("Exactitude Zero-R", f"{max(p_emp, 1-p_emp):.1%}")

    st.markdown(
        f"""
<div class="dm-card">
  <b style="color:#ffffff;font-size:14px">Mise en perspective avec le jeu de test réel :</b><br>
  <span style="color:#cbd5e1;font-size:13px">
  Sur les 58 766 candidats réels, la classe <code>Employed = 1</code> représente 53,6 %.<br>
  Zero-R prédit toujours 1, ce qui lui donne mécaniquement <b>53,6 % d'accuracy</b> mais un <b>F1-macro de seulement 34,9 %</b>.<br>
  Dans le scénario « Profil seul » (sans technologies), tous les modèles complexes chutent à ≈ 54–58 % d'accuracy :
  ils ne battent pas substantiellement cette baseline triviale, confirmant l'absence de signal utile dans les variables de profil.
  </span>
</div>
""",
        unsafe_allow_html=True
    )
    return majority_class


# ──────────────────────────────────────────────────────────────────────────────
# Synthèse Comparative & Performance Globale
# ──────────────────────────────────────────────────────────────────────────────
def _render_comparison_table(preds_dict: dict, true_label: int | None):
    st.markdown("---")
    st.markdown("### 📊 Matrice d'Arbitrage Croisé des 6 Classifieurs")

    rows = []
    for algo_id, (algo_label, pred_val) in preds_dict.items():
        if pred_val is None:
            continue
        row_info = {
            "Algorithme": algo_label,
            "Prédiction du modèle": "Employé (1)" if pred_val == 1 else "Non employé (0)",
        }
        if true_label is not None:
            row_info["Vraie cible dataset"] = "Employé (1)" if true_label == 1 else "Non employé (0)"
            row_info["Adéquation"] = "✅ Conforme" if pred_val == true_label else "❌ Divergence"
        rows.append(row_info)

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    valid_p = [p for _, (_, p) in preds_dict.items() if p is not None]
    if valid_p:
        n_emp = valid_p.count(1)
        consensus_cls = 1 if n_emp >= len(valid_p) / 2 else 0
        consensus_pct = (n_emp if consensus_cls == 1 else (len(valid_p) - n_emp)) / len(valid_p)
        clr = CLR_EMPLOYED if consensus_cls == 1 else CLR_NOT_EMP

        st.markdown(
            f"""
<div class="dm-card" style="border-left:4px solid {clr} !important">
  <b style="color:{clr};font-size:15px">Consensus d'Ensemble ({consensus_pct:.0%}) ➔ {'Employé (1)' if consensus_cls == 1 else 'Non employé (0)'}</b><br>
  <span style="color:#cbd5e1;font-size:13px">
    {n_emp} modèle(s) sur {len(valid_p)} convergent vers la classe « Employé ».
    {"Tous les modèles s'accordent rigoureusement." if consensus_pct == 1.0 else "Des divergences locales se manifestent selon la topologie des frontières."}
  </span>
</div>
""",
            unsafe_allow_html=True
        )


def _render_global_evaluation(use_full: bool):
    st.markdown("---")
    st.markdown("### 📈 Performances Globales Réelles (Jeu de Test Séparé, N = 14 692)")
    st.caption("Mesures calculées et enregistrées lors des évaluations du notebook 03.")

    t1, t2, t3 = st.tabs(["🚀 Toutes les variables", "🔬 Profil seul (sans technologies)", "🔍 Diagnostic de la Règle"])

    with t1:
        res = _load_csv("results.csv")
        if res is not None:
            st.dataframe(res.round(4), width="stretch", hide_index=True)

            if "Modèle" in res.columns and "F1 macro" in res.columns:
                fig, ax = plt.subplots(figsize=(6, 2.6), facecolor=CLR_BG)
                ax.set_facecolor("#0b1329")
                mods = res["Modèle"].tolist()
                f1_vals = res["F1 macro"].fillna(0).tolist()
                colors = [CLR_EMPLOYED if v > 0.9 else (CLR_CANDIDATE if v > 0.7 else CLR_NOT_EMP) for v in f1_vals]
                ax.barh(mods[::-1], f1_vals[::-1], color=colors[::-1], alpha=0.9)
                ax.set_xlim(0, 1.05)
                ax.set_xlabel("F1-macro sur le jeu de test", color="#cbd5e1", fontsize=8)
                ax.tick_params(colors="#94a3b8", labelsize=8)
                ax.spines[:].set_color(CLR_BORDER)
                fig.tight_layout(pad=0.3)
                st.pyplot(fig)
                plt.close(fig)

    with t2:
        prof = _load_csv("profile_only_results.csv")
        if prof is not None:
            st.dataframe(prof.round(4), width="stretch", hide_index=True)
            st.warning(
                "⚠️ **Preuve d'inutilité des profils sans technologies :** En neutralisant `HaveWorkedWith` et `ComputerSkills`, "
                "aucun modèle ne dépasse ≈ 58 % d'accuracy (seuil Zero-R = 53,6 %). Les attributs démographiques sont inopérants."
            )

    with t3:
        diag = _load_csv("diagnostic.csv")
        if diag is not None:
            st.dataframe(diag, width="stretch", hide_index=True)
            fig_d, ax_d = plt.subplots(figsize=(6, 2.6), facecolor=CLR_BG)
            ax_d.set_facecolor("#0b1329")
            nk = diag["n_key"].values
            e0 = diag["Employed=0"].values
            e1 = diag["Employed=1"].values
            w = 0.38
            ax_d.bar(nk - w/2, e0, w, color=CLR_NOT_EMP, label="Employed = 0", alpha=0.85)
            ax_d.bar(nk + w/2, e1, w, color=CLR_EMPLOYED, label="Employed = 1", alpha=0.85)
            ax_d.axvline(2.5, color=CLR_CANDIDATE, lw=2, linestyle="--", label="Frontière stricte (≥ 3)")
            ax_d.set_xlabel("Nombre de technologies clés (0 à 10)", color="#cbd5e1", fontsize=8)
            ax_d.set_ylabel("Effectif candidats", color="#cbd5e1", fontsize=8)
            ax_d.tick_params(colors="#94a3b8", labelsize=8)
            ax_d.spines[:].set_color(CLR_BORDER)
            ax_d.legend(fontsize=7, facecolor=CLR_CARD, labelcolor="#cbd5e1")
            fig_d.tight_layout(pad=0.3)
            st.pyplot(fig_d)
            plt.close(fig_d)


# ──────────────────────────────────────────────────────────────────────────────
# Point d'Entrée Principal appelé depuis src/app.py
# ──────────────────────────────────────────────────────────────────────────────
def render_simulation():
    _inject_custom_css()

    st.markdown(
        f'<h2 style="color:#ffffff;display:flex;align-items:center;gap:10px">'
        f'<span>🧪</span> Laboratoire d\'Analyse & Simulation des Algorithmes de Classification'
        f'</h2>',
        unsafe_allow_html=True
    )
    st.caption("Comparaison analytique des modèles : KNN, Arbre CART, Linear SVC, Naive Bayes, Random Forest et Zero-R.")

    # 1. Audit Déontologique & Gouvernance (Format Professionnel)
    _render_governance_report()

    # 2. Sélection du Candidat (Personas, Bouton Aléatoire, Manuel)
    meta = _load_meta()
    champion_pipe = _load_champion_pipeline()
    candidate_df, use_full, true_label = _render_candidate_selection(meta)

    # 3. Simulateur Canvas 2D Interactif (Positionné en HAUT pour Exploration Immédiate)
    _render_interactive_canvas_sandbox()

    # 4. Ajustement des Modèles
    with st.spinner("Ajustement des modèles interactifs sur l'échantillon de travail..."):
        fitted_models, X_tr, y_tr, X_te, y_te = _fit_simulation_models(use_full)

    # 5. Onglets des Algorithmes
    st.markdown("---")
    st.markdown("### 🔬 Exploration Analytique Détaillée par Algorithme")

    tabs = st.tabs([
        "📐 Linear SVC (Champion)",
        "🎯 KNN (Voisinage)",
        "🌳 Arbre de Décision",
        "🧮 Naive Bayes",
        "🌲 Random Forest",
        "⚖️ Zero-R (Baseline)"
    ])

    predictions = {}

    with tabs[0]:
        p_svc = _render_svm_view(candidate_df, use_full, champion_pipe, X_tr, y_tr, true_label)
        predictions["svc"] = ("📐 Linear SVC", p_svc)

    with tabs[1]:
        p_knn = _render_knn_view(candidate_df, use_full, X_tr, y_tr, true_label)
        predictions["knn"] = ("🎯 K-Nearest Neighbors", p_knn)

    with tabs[2]:
        p_tree = _render_tree_view(candidate_df, use_full, X_tr, y_tr, true_label)
        predictions["tree"] = ("🌳 Arbre de Décision", p_tree)

    with tabs[3]:
        p_nb = _render_nb_view(candidate_df, use_full, X_tr, y_tr, true_label)
        predictions["nb"] = ("🧮 Naive Bayes", p_nb)

    with tabs[4]:
        p_rf = _render_rf_view(candidate_df, use_full, X_tr, y_tr, true_label)
        predictions["rf"] = ("🌲 Random Forest", p_rf)

    with tabs[5]:
        p_zr = _render_zeror_view(y_tr)
        predictions["zeror"] = ("⚖️ Zero-R (Baseline)", p_zr)

    # 6. Synthèse Comparative & Consensus
    _render_comparison_table(predictions, true_label)

    # 7. Performances Globales Réelles
    _render_global_evaluation(use_full)
