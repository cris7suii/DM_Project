"""Application Streamlit — Démonstrateur Didactique d'Employabilité des Candidats.

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

KEY_TECHS = [
    "Microsoft SQL Server", "npm", "jQuery", "TypeScript", "MongoDB",
    "SQLite", "Bash/Shell", "Node.js", "C#", "Java"
]

st.set_page_config(
    page_title="EmployML · Classification & Analyse d'Employabilité",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed",
)
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
    st.error("⚠️ Modèle introuvable : veuillez exécuter les notebooks 01, 02 et 03 (dossier `notebooks/`).")
    st.stop()

tabs = st.tabs([
    "🏠 Accueil",
    "🔮 Inférence & Prédiction",
    "📊 Benchmarking & Performance",
    "🔎 Analyse d'Ablations",
    "⚖️ Équité & IA Responsable",
    "🧪 Laboratoire de Simulation",
    "ℹ️ Architecture & À propos"
])

# ------------------------------------------------------------------ 🏠 Accueil
with tabs[0]:
    left, right = st.columns([3, 2], gap="large")
    with left:
        hero(
            eyebrow="Projet Académique · Data Mining & Machine Learning",
            line1="Prédire l'employabilité",
            line2="à partir du profil des candidats.",
            copy="Plateforme d'exploration et d'audit algorithmique appliquée au jeu de données "
                 "Kaggle « 70k+ Job Applicants Data » (73 462 profils). Analyse comparative de modèles "
                 "de Machine Learning classique et mise en lumière des biais déterministes sous-jacents."
        )
    with right:
        warning_card(
            title="Découverte Majeure : Règle Déterministe Cachée",
            text="L'audit approfondi (notebook 03) révèle que la cible « Employed » est une fonction logique "
                 "exacte : un candidat est classé 1 si et seulement s'il maîtrise au moins 3 technologies "
                 "clés parmi 10. Les performances parfaites proviennent de cette règle et non d'une véritable mesure d'employabilité."
        )

    st.write("")
    res_home = load_csv("results.csv")
    n_models = len(res_home) if res_home is not None else 6
    metric_cards([
        {"icon": "🏆", "value": meta.get("champion", "Linear SVC"), "label": "Modèle Champion", "hint": "Sélectionné sur F1-macro (3 plis)", "tone": "violet"},
        {"icon": "🗄️", "value": f'{meta.get("n_train", 58766):,}'.replace(",", " "), "label": "Candidats en Entraînement", "hint": "Split stratifié 80/20", "tone": "indigo"},
        {"icon": "👥", "value": f'{meta.get("target_rate", 0.536):.1%}', "label": "Taux d'Emploi Cible", "hint": "Classe majoritaire (Employed)", "tone": "green"},
        {"icon": "🧠", "value": str(n_models), "label": "Algorithmes Évalués", "hint": "ML classique scikit-learn", "tone": "cyan"},
    ])
    
    st.write("")
    pipeline([
        ("Nettoyage & EDA", "Gestion des valeurs manquantes, doublons et corrections d'incohérences"),
        ("Prétraitement", "Pipelines scikit-learn : encodage One-Hot, scaling et multi-hot 40 techs"),
        ("Optimisation", "GridSearchCV avec 3 plis stratifiés sur sous-échantillon représentatif"),
        ("Évaluation & Éthique", "Ablations multivariées, audit d'équité et explicabilité des décisions"),
    ])

# ------------------------------------------------------------------ 🔮 Prédiction
with tabs[1]:
    page_header(
        eyebrow="Inférence en Temps Réel",
        title="Simulateur de Profil & Prédiction du Candidat",
        description="Configurez les caractéristiques d'un profil pour observer la prédiction générée par le modèle champion."
    )
    
    form_col, res_col = st.columns([3, 2], gap="large")
    
    with form_col:
        with st.form("profil_candidat"):
            st.markdown("##### 👤 1. Identité & Profil Démographique")
            col_id1, col_id2, col_id3 = st.columns(3)
            age = col_id1.selectbox("Tranche d'âge", ["<35", ">35"], help="Âge du candidat")
            gender = col_id2.selectbox("Genre", ["Man", "Woman", "NonBinary"], help="Genre déclaré")
            country = col_id3.selectbox(
                "Pays de résidence",
                meta["countries"],
                index=meta["countries"].index("Germany") if "Germany" in meta["countries"] else 0,
                help="Pays du candidat (les pays rares sont regroupés)"
            )

            st.markdown("##### 🎓 2. Formation & Expérience Professionnelle")
            col_exp1, col_exp2, col_exp3 = st.columns(3)
            edlevel = col_exp1.selectbox("Niveau d'études", ["Undergraduate", "Master", "PhD", "NoHigherEd", "Other"])
            mainbranch = col_exp2.selectbox("Type d'activité", ["Dev", "NotDev"], help="Développeur professionnel ou profil transverse")
            employment = col_exp3.selectbox("Statut Employment déclaratif", [1, 0], help="Indicateur d'emploi brut du questionnaire")

            col_sal1, col_sal2, col_sal3 = st.columns(3)
            years_code = col_sal1.number_input("Années de pratique du code", 0, 50, 10)
            years_pro = col_sal2.number_input("Années d'expérience professionnelle", 0, 50, 5)
            salary = col_sal3.number_input("Dernier salaire annuel (USD)", 1000, 500000, 65000, step=2500)

            st.markdown("##### 💻 3. Stack Technique & Compétences Clés")
            techs = st.multiselect(
                "Technologies maîtrisées (parmi les 40 les plus fréquentes)",
                meta["technologies"],
                default=["Python", "Git", "TypeScript"],
                help="Sélectionnez les technologies du candidat. Certaines font partie de la liste clé."
            )
            others = st.number_input("Nombre d'autres technologies maîtrisées (hors liste)", 0, 100, 3)

            st.markdown("##### 🛡️ 4. Facteurs d'Inclusion & Diversité")
            col_eq1, col_eq2 = st.columns(2)
            mental = col_eq1.selectbox("Santé mentale (déclaration)", ["No", "Yes"], help="Variable sensible présente dans le dataset")
            access = col_eq2.selectbox("Accessibilité / Handicap", ["No", "Yes"], help="Besoin d'aménagements d'accessibilité")

            st.write("")
            submit = st.form_submit_button("✨ Calculer la Prédiction & Diagnostiquer")

    with res_col:
        if submit:
            effective_years_pro = min(years_pro, years_code)
            total_skills = len(techs) + int(others)
            
            row = pd.DataFrame([{
                "Age": age,
                "Accessibility": access,
                "EdLevel": edlevel,
                "Employment": employment,
                "Gender": gender,
                "MentalHealth": mental,
                "MainBranch": mainbranch,
                "YearsCode": years_code,
                "YearsCodePro": effective_years_pro,
                "Country": country,
                "PreviousSalary": float(salary),
                "HaveWorkedWith": ";".join(techs),
                "ComputerSkills": total_skills,
            }])
            
            pred = int(model.predict(row)[0])
            clf = model.named_steps["clf"]
            
            proba = None
            score_text = ""
            if hasattr(clf, "predict_proba"):
                proba = float(model.predict_proba(row)[0, 1])
                score_text = f"Probabilité estimée : <strong>{proba:.1%}</strong>"
            elif hasattr(model, "decision_function"):
                dist = float(model.decision_function(row)[0])
                score_text = f"Distance à l'hyperplan : <strong>{dist:+.2f}</strong> (>0 : classe Employé)"
                # Approximation de confiance logistique pour la jauge
                approx_proba = 1.0 / (1.0 + (2.71828 ** (-dist)))
                proba = approx_proba

            key_techs_present = [t for t in techs if t in KEY_TECHS]
            result_card(pred, score_text, proba, key_techs_found=key_techs_present)
            
            st.caption(
                "ℹ️ **Note académique** : Cette prédiction reflète la reproduction d'une règle synthétique "
                "sur les technologies déclarées et ne saurait constituer une évaluation humaine de compétence."
            )
        else:
            st.info(
                "👈 Ajustez les paramètres du candidat dans le formulaire de gauche, "
                "puis cliquez sur **« ✨ Calculer la Prédiction & Diagnostiquer »**."
            )

# ------------------------------------------------------------------ 📊 Performance
with tabs[2]:
    page_header(
        eyebrow="Benchmarking des Algorithmes",
        title="Comparaison Multi-Modèles sur le Jeu de Test",
        description="Résultats comparés des 6 architectures de classification évaluées sur des données jamais vues lors de l'entraînement."
    )
    
    res = load_csv("results.csv")
    if res is not None:
        champ_row = res.sort_values(by="F1 macro", ascending=False).iloc[0]
        base_row = res[res["Modèle"].str.contains("Zero-R", case=False, na=False)]
        base_f1 = base_row["F1 macro"].values[0] if not base_row.empty else 0.3490
        gain_f1 = (champ_row["F1 macro"] - base_f1) * 100
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Modèle Champion", champ_row["Modèle"], f"F1: {champ_row['F1 macro']:.4f}")
        col_m2.metric("Baseline (Zero-R)", f"{base_f1:.4f}", "Seuil d'utilité")
        col_m3.metric("Gain Absolu vs Baseline", f"+{gain_f1:.1f} pts", "F1-score")

        st.write("")
        t_col, c_col = st.columns([3, 2], gap="large")
        with t_col:
            st.markdown("##### Tableau des Performances (Test Set)")
            st.dataframe(res.round(4), width="stretch", hide_index=True)
            st.caption("Critère officiel de sélection : **F1-macro en validation croisée 3 plis** (pour éviter tout surapprentissage du test set).")
        with c_col:
            st.markdown("##### Comparatif F1-macro vs Accuracy")
            chart_data = res.set_index("Modèle")[["F1 macro", "Accuracy"]]
            st.bar_chart(chart_data, color=["#6366F1", "#06B6D4"])

        if res["F1 macro"].max() > 0.99:
            st.write("")
            warning_card(
                title="Performance Parfaite (≈ 100 %) : Explication de la Fuite Déterministe",
                text="Les modèles linéaires (Linear SVC) et arborescents retrouvent exactement la partition "
                     "logique induite par les colonnes de technologies. Ce résultat illustre un biais fréquent "
                     "dans les jeux de données synthétiques Kaggle."
            )

    st.write("")
    diag_col, prof_col = st.columns(2, gap="large")
    with diag_col:
        diag = load_csv("diagnostic.csv")
        if diag is not None:
            st.markdown("##### 🎯 Diagnostic de la Règle : Nombre de Clés vs % Employé")
            st.dataframe(diag, width="stretch", hide_index=True)
            st.caption("Bascule nette : dès 3 technologies clés maîtrisées, le taux d'emploi passe instantanément à 100 %.")
            
    with prof_col:
        prof = load_csv("profile_only_results.csv")
        if prof is not None:
            st.markdown("##### 📉 Performance Réelle : Profil Seul (Sans Technologies)")
            st.dataframe(prof.round(4), width="stretch", hide_index=True)
            st.caption("Sans les technologies, tous les algorithmes s'effondrent à ≈ 57 % (niveau du hasard biaisé de la baseline Zero-R à 53.6 %).")

# ------------------------------------------------------------------ 🔎 Ablations
with tabs[3]:
    page_header(
        eyebrow="Analyse d'Importance & Robustesse",
        title="Étude d'Ablations : D'où Vient la Performance ?",
        description="Mesure de l'impact causé par le retrait progressif de sous-ensembles de caractéristiques sur le score F1-macro."
    )
    
    ab = load_csv("ablation.csv")
    if ab is not None:
        t_col, c_col = st.columns([3, 2], gap="large")
        with t_col:
            st.markdown("##### Scénarios d'Ablation Testés")
            st.dataframe(ab.round(4), width="stretch", hide_index=True)
            st.caption("Le retrait de `HaveWorkedWith` provoque une chute immédiate de plus de 42 points de F1-macro.")
        with c_col:
            st.markdown("##### Chute de Performance par Scénario")
            st.bar_chart(ab.set_index("Scénario")[["F1 macro"]], color=["#8B5CF6"])
            
        st.write("")
        st.markdown(
            "> **Enseignement méthodologique** : L'expérience d'ablation prouve que ni les diplômes, "
            "ni le salaire, ni l'expérience professionnelle n'expliquent la cible. Tout le pouvoir "
            "prédictif est concentré dans les chaînes textuelles de technologies."
        )

# ------------------------------------------------------------------ ⚖️ Équité
with tabs[4]:
    page_header(
        eyebrow="Éthique & Audit Algorithmique",
        title="Audit d'Équité par Attribut Sensible",
        description="Évaluation de la parité démographique et de l'égalité des chances sur le genre, l'âge, la santé mentale et le handicap."
    )
    
    ethical_card(
        title="Corrélation n'est pas Qualification",
        text="Les disparités observées entre sous-groupes démographiques découlent des déséquilibres "
             "de représentation dans les technologies déclarées. Un modèle d'aide au recrutement basé sur "
             "ces données reproduirait et amplifierait ces disparités de manière injustifiée."
    )
    
    fair = load_csv("fairness.csv")
    if fair is not None:
        st.markdown("##### Métriques d'Équité Détaillées par Sous-Groupe")
        st.dataframe(fair.round(3), width="stretch", hide_index=True)
        st.caption("`taux prédit` = taux de sélection positive ; `TPR` = True Positive Rate (Rappel) ; `FPR` = False Positive Rate.")

# ------------------------------------------------------------------ 🧪 Simulation
with tabs[5]:
    render_simulation()

# ------------------------------------------------------------------ ℹ️ À propos
with tabs[6]:
    page_header(
        eyebrow="Ressources & Gouvernance",
        title="À Propos du Projet EmployML",
        description="Fiche technique, architecture logicielle et protocole expérimental."
    )
    
    st.markdown(
        badge("Python 3.11", "champion") + " " +
        badge("scikit-learn 1.5+", "champion") + " " +
        badge("pandas 2.2+", "ok") + " " +
        badge("SciPy 1.11+", "ok") + " " +
        badge("Streamlit 1.50+", "champion") + " " +
        badge("Kaggle 70k+", "ok"),
        unsafe_allow_html=True
    )
    st.write("")
    
    col_about1, col_about2 = st.columns(2, gap="large")
    with col_about1:
        st.markdown("##### 🛠️ Stack Technologique & Contraintes")
        st.markdown(
            "- **Machine Learning Classique** : Implémenté exclusivement avec Scikit-Learn (aucun Deep Learning, respect strict du cahier des charges).\n"
            "- **Préprocesseur Robuste** : `ColumnTransformer` hermétique ré-ajusté à chaque pli de validation croisée.\n"
            "- **Reproductibilité** : Fixation stricte de toutes les graines pseudo-aléatoires (`random_state=42`).\n"
            "- **Tests Automatisés** : Suite complète de tests unitaires validant l'intégrité de la simulation et des artefacts."
        )
    with col_about2:
        st.markdown("##### 🎓 Objectifs Pédagogiques du Module")
        st.markdown(
            "- **Détection des Règles Déterministes** : Identification des artefacts synthétiques dans les jeux de données ouverts.\n"
            "- **Transparence & Interprétabilité** : Décomposition visuelle des hyperplans (SVC), sphéroïdes (KNN) et arbres (CART).\n"
            "- **Sensibilisation à l'IA Responsable** : Mesure rigoureuse des biais d'équité et de représentativité.\n"
            "- **Déploiement Cloud** : Prêt pour le déploiement continu via Render (`render.yaml`)."
        )
