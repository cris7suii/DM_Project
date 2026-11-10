"""Application Streamlit — démo de la classification de l'employabilité des candidats.

Lancer depuis la racine du projet :  streamlit run src/app.py
Prérequis : avoir exécuté les notebooks 01 -> 03 (génère models/best_model.pkl, results.csv, ...).
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))          # nécessaire pour désérialiser split_techs / _log1p
import features  # noqa: E402,F401
from simulation import build_html  # noqa: E402
from simulation_v2 import render_simulation  # noqa: E402
from theme import (  # noqa: E402
    badge, ethical_card, hero, inject_theme, metric_cards,
    page_header, pipeline, result_card, warning_card,
)

st.set_page_config(page_title="EmployML · Employabilité des candidats", page_icon="💼", layout="wide")
inject_theme()


@st.cache_resource
def load_artifacts():
    model = joblib.load(ROOT / "models" / "best_model.pkl")
    meta = json.load(open(ROOT / "models" / "meta.json", encoding="utf-8"))
    return model, meta


@st.cache_data
def load_csv(name):
    p = ROOT / "models" / name
    return pd.read_csv(p) if p.exists() else None


try:
    model, meta = load_artifacts()
except FileNotFoundError:
    st.error("Modèle introuvable : exécutez d'abord les notebooks 01, 02 et 03 (dossier `notebooks/`).")
    st.stop()

tabs = st.tabs(["🏠 Accueil", "🔮 Prédiction", "📊 Performance", "🔎 Ablations", "⚖️ Équité", "🧪 Simulation", "ℹ️ À propos"])

# ------------------------------------------------------------------ Accueil
with tabs[0]:
    left, right = st.columns([3, 2], gap="large")
    with left:
        hero("Data Mining · Projet académique",
             "Prédire l'employabilité",
             "à partir du profil des candidats.",
             "Explorez comment des modèles de Machine Learning classique classent les candidats selon leur profil "
             "(formation, expérience, pays) et les technologies qu'ils maîtrisent. Jeu de données : 70k+ Job Applicants Data.")
    with right:
        warning_card("La cible est une règle déterministe",
                     "« Employed » = maîtriser au moins 3 technologies parmi 10. "
                     "Ce n'est pas une mesure de l'employabilité réelle.")

    st.write("")
    res_home = load_csv("results.csv")
    n_models = len(res_home) if res_home is not None else 6
    metric_cards([
        {"icon": "🏆", "value": meta["champion"], "label": "Modèle champion", "hint": "Choisi sur F1-macro (CV)", "tone": "violet"},
        {"icon": "🗄️", "value": f'{meta["n_train"]:,}'.replace(",", " "), "label": "Candidats (train)", "hint": "Split stratifié 80/20", "tone": "indigo"},
        {"icon": "👥", "value": f'{meta["target_rate"]:.1%}', "label": "Part d'employés", "hint": "Taux de la cible", "tone": "green"},
        {"icon": "🧠", "value": str(n_models), "label": "Modèles comparés", "hint": "ML classique, sans Deep Learning", "tone": "cyan"},
    ])
    st.write("")
    pipeline([("Nettoyage", "Valeurs manquantes, doublons"), ("Prétraitement", "Encodage & standardisation"),
              ("Modèles", "GridSearchCV, 3 plis"), ("Évaluation", "Ablations & équité")])

# ------------------------------------------------------------------ Prédiction
with tabs[1]:
    page_header("Inférence en direct", "Prédiction d'un candidat",
                "Saisissez un profil pour obtenir la classification du modèle champion.")
    form_col, res_col = st.columns([3, 2], gap="large")
    with form_col:
        with st.form("profil"):
            a, b, c = st.columns(3)
            age = a.selectbox("Âge", ["<35", ">35"])
            edlevel = a.selectbox("Niveau d'études", ["Undergraduate", "Master", "PhD", "NoHigherEd", "Other"])
            mainbranch = a.selectbox("Profil", ["Dev", "NotDev"])
            years_code = b.number_input("Années de pratique du code", 0, 50, 10)
            years_pro = b.number_input("Années d'expérience professionnelle", 0, 50, 5)
            salary = b.number_input("Salaire précédent (USD/an)", 1, 300000, 60000, step=1000)
            country = c.selectbox("Pays", meta["countries"],
                                  index=meta["countries"].index("Germany") if "Germany" in meta["countries"] else 0)
            employment = c.selectbox("Employment (indicateur du jeu de données)", [1, 0])
            gender = c.selectbox("Genre", ["Man", "Woman", "NonBinary"])
            mental = a.selectbox("MentalHealth", ["No", "Yes"])
            access = b.selectbox("Accessibility", ["No", "Yes"])
            techs = st.multiselect("Technologies maîtrisées (les plus fréquentes)", meta["technologies"],
                                   default=["Python", "Git"])
            others = st.number_input("Autres technologies maîtrisées (hors liste ci-dessus)", 0, 100, 3)
            go = st.form_submit_button("✨ Lancer la prédiction")

    with res_col:
        if go:
            row = pd.DataFrame([{
                "Age": age, "Accessibility": access, "EdLevel": edlevel, "Employment": employment,
                "Gender": gender, "MentalHealth": mental, "MainBranch": mainbranch,
                "YearsCode": years_code, "YearsCodePro": min(years_pro, years_code),
                "Country": country, "PreviousSalary": float(salary),
                "HaveWorkedWith": ";".join(techs), "ComputerSkills": len(techs) + int(others),
            }])
            pred = int(model.predict(row)[0])
            clf = model.named_steps["clf"]
            proba = None
            if hasattr(clf, "predict_proba"):
                proba = float(model.predict_proba(row)[0, 1])
                score = f"{proba:.1%} de probabilité estimée d'être employé"
            elif hasattr(model, "decision_function"):
                score = f'Score de décision = {float(model.decision_function(row)[0]):+.2f} (>0 : classe « employé »)'
            else:
                score = ""
            result_card(pred, score, proba)
            st.caption("Démo académique : la prédiction est indicative et ne doit jamais servir à décider "
                       "d'un recrutement. Voir l'onglet « Équité ».")
        else:
            st.info("Remplissez le profil à gauche puis lancez la prédiction.")

# ------------------------------------------------------------------ Performance
with tabs[2]:
    page_header("Évaluation des modèles", "Comparaison des modèles", "Résultats sur le jeu de test.")
    res = load_csv("results.csv")
    if res is not None:
        t_col, c_col = st.columns([3, 2], gap="large")
        with t_col:
            st.dataframe(res.round(4), width="stretch", hide_index=True)
            st.caption("Le champion est choisi sur le F1-macro en validation croisée (pas sur le test).")
        with c_col:
            st.bar_chart(res.set_index("Modèle")[["F1 macro", "Accuracy"]], color=["#6366F1", "#22D3EE"])
        if res["F1 macro"].max() > 0.99:
            warning_card("Score ≈ 100 % : ce n'est pas un exploit",
                         "Le diagnostic (notebook 03, §6) montre que la cible `Employed` est une règle déterministe sur la "
                         "liste de technologies `HaveWorkedWith` (≥ k technologies « clés »). "
                         "Voir ci-dessous la comparaison sans ces variables.")
    diag = load_csv("diagnostic.csv")
    if diag is not None:
        st.subheader("Diagnostic : nombre de technologies « clés » maîtrisées vs Employed")
        st.dataframe(diag, width="stretch", hide_index=True)
    prof = load_csv("profile_only_results.csv")
    if prof is not None:
        st.subheader("Comparaison sans technologies ni ComputerSkills (profil seul)")
        st.dataframe(prof.round(4), width="stretch", hide_index=True)
        st.caption("Proches de la baseline Zero-R : formation, pays, expérience, salaire et attributs sensibles "
                   "portent très peu d'information sur la cible.")

# ------------------------------------------------------------------ Ablations
with tabs[3]:
    page_header("Analyse de dépendance", "Ablations : d'où vient la performance ?",
                "Score obtenu après retrait de groupes de variables.")
    ab = load_csv("ablation.csv")
    if ab is not None:
        t_col, c_col = st.columns([3, 2], gap="large")
        with t_col:
            st.dataframe(ab.round(4), width="stretch", hide_index=True)
        with c_col:
            st.bar_chart(ab.set_index("Scénario")[["F1 macro"]], color=["#6366F1"])

# ------------------------------------------------------------------ Équité
with tabs[4]:
    page_header("IA responsable", "Analyse d'équité par attribut sensible",
                "Comportement du modèle selon le genre, l'âge, la santé mentale et l'accessibilité.")
    ethical_card("Corrélation n'est pas qualification.",
                 "La cible est construite à partir des technologies déclarées. Les écarts entre groupes ne mesurent pas "
                 "l'employabilité réelle et ne doivent jamais servir à décider d'un recrutement.")
    fair = load_csv("fairness.csv")
    if fair is not None:
        st.dataframe(fair.round(3), width="stretch", hide_index=True)
        st.caption("`taux prédit` = part de candidats prédits « employés » ; `TPR`/`FPR` = taux de vrais/faux positifs par groupe.")

# ------------------------------------------------------------------ Simulation
with tabs[5]:
    render_simulation()

# ------------------------------------------------------------------ À propos
with tabs[6]:
    page_header("Informations", "À propos", "Stack technique et limites du projet.")
    st.markdown(badge("Python", "champion") + " " + badge("pandas") + " " + badge("scikit-learn") + " "
                + badge("SciPy") + " " + badge("Streamlit"), unsafe_allow_html=True)
    st.write("")
    st.write("**Stack** : Python · pandas · NumPy · scikit-learn · SciPy · GridSearchCV · joblib · Streamlit. "
             "Aucune bibliothèque de Deep Learning.")
    st.write("Projet académique : les prédictions sont indicatives. Les attributs sensibles (genre, âge, santé mentale, "
             "accessibilité) présents dans le jeu de données posent des questions éthiques discutées dans le rapport.")
