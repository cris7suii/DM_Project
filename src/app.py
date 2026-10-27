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

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))          # nécessaire pour désérialiser split_techs / _log1p
import features  # noqa: E402,F401

st.set_page_config(page_title="Employabilité des candidats", page_icon="💼", layout="wide")


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

tabs = st.tabs(["🏠 Accueil", "🔮 Prédiction", "📊 Performance", "🔎 Ablations", "⚖️ Équité", "ℹ️ À propos"])

# ------------------------------------------------------------------ Accueil
with tabs[0]:
    st.title("💼 Classification des candidats selon leur employabilité")
    st.write(
        "Projet de **Data Mining / Machine Learning classique** (sans Deep Learning) : prédire si un candidat "
        "est en situation d'emploi (`Employed`) à partir de son profil (formation, expérience, technologies maîtrisées, "
        "pays, …). Jeu de données : *70k+ Job Applicants Data*."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Modèle champion", meta["champion"])
    c2.metric("Candidats (train)", f'{meta["n_train"]:,}'.replace(",", " "))
    c3.metric("Part d'employés", f'{meta["target_rate"]:.1%}')

# ------------------------------------------------------------------ Prédiction
with tabs[1]:
    st.header("Prédiction en direct")
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
        go = st.form_submit_button("Prédire")

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
        if hasattr(clf, "predict_proba"):
            score = f'{model.predict_proba(row)[0, 1]:.1%} de probabilité estimée d\'être employé'
        elif hasattr(model, "decision_function"):
            score = f'score de décision = {float(model.decision_function(row)[0]):+.2f} (>0 : classe « employé »)'
        else:
            score = ""
        (st.success if pred == 1 else st.warning)(
            f'Prédiction : **{"Employé (1)" if pred == 1 else "Non employé (0)"}** — {score}')
        st.caption("Démo académique : la prédiction est indicative et ne doit jamais servir à décider "
                   "d'un recrutement. Voir l'onglet « Équité ».")

# ------------------------------------------------------------------ Performance
with tabs[2]:
    st.header("Comparaison des modèles (jeu de test)")
    res = load_csv("results.csv")
    if res is not None:
        st.dataframe(res.round(4), width="stretch", hide_index=True)
        st.bar_chart(res.set_index("Modèle")[["F1 macro", "Accuracy"]])
        st.caption("Le champion est choisi sur le F1-macro en validation croisée (pas sur le test).")
        if res["F1 macro"].max() > 0.99:
            st.warning(
                "⚠️ Score ≈ 100 % : ce n'est pas un exploit du modèle. Le diagnostic (notebook 03, §6) montre que la "
                "cible `Employed` est une règle déterministe sur la liste de technologies `HaveWorkedWith` "
                "(≥ k technologies « clés »). Voir ci-dessous la comparaison sans ces variables.")
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
    st.header("Ablations : d'où vient la performance ?")
    ab = load_csv("ablation.csv")
    if ab is not None:
        st.dataframe(ab.round(4), width="stretch", hide_index=True)
        st.bar_chart(ab.set_index("Scénario")[["F1 macro"]])

# ------------------------------------------------------------------ Équité
with tabs[4]:
    st.header("Analyse d'équité par attribut sensible")
    fair = load_csv("fairness.csv")
    if fair is not None:
        st.dataframe(fair.round(3), width="stretch", hide_index=True)
        st.caption("`taux prédit` = part de candidats prédits « employés » ; `TPR`/`FPR` = taux de vrais/faux positifs par groupe.")

# ------------------------------------------------------------------ À propos
with tabs[5]:
    st.header("À propos")
    st.write("**Stack** : Python · pandas · NumPy · scikit-learn · SciPy · GridSearchCV · joblib · Streamlit. "
             "Aucune bibliothèque de Deep Learning.")
    st.write("Projet académique : les prédictions sont indicatives. Les attributs sensibles (genre, âge, santé mentale, "
             "accessibilité) présents dans le jeu de données posent des questions éthiques discutées dans le rapport.")
