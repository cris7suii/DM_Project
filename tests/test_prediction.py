"""Tests de prédiction, d'inférence ML, de la règle déterministe et des cas limites."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import candidate
from cli import predict_single


# ==============================================================================
# Tests de dimension et de validité des prédictions
# ==============================================================================

def test_single_prediction_shape_and_classes(loaded_artifacts, sample_candidate_df: pd.DataFrame):
    """Vérifie que la prédiction sur un profil unique renvoie une classe binaire valide."""
    model, _ = loaded_artifacts
    pred = model.predict(sample_candidate_df)

    assert isinstance(pred, np.ndarray)
    assert pred.shape == (1,)
    assert pred[0] in (0, 1), f"Classe invalide retournée : {pred[0]}"


def test_batch_prediction_shape_and_classes(loaded_artifacts, sample_batch_df: pd.DataFrame):
    """Vérifie que la prédiction sur un lot de profils renvoie la dimension exacte."""
    model, _ = loaded_artifacts
    preds = model.predict(sample_batch_df)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(sample_batch_df),)
    assert set(preds).issubset({0, 1})


def test_decision_function_sign_consistency(loaded_artifacts, sample_batch_df: pd.DataFrame):
    """Vérifie la cohérence entre le signe de la marge Linear SVC et la classe prédite."""
    model, _ = loaded_artifacts
    preds = model.predict(sample_batch_df)
    scores = model.decision_function(sample_batch_df)

    assert scores.shape == (len(sample_batch_df),)
    for pred, score in zip(preds, scores):
        if score > 0:
            assert pred == 1, f"Score positif ({score}) doit prédire la classe 1"
        elif score < 0:
            assert pred == 0, f"Score négatif ({score}) doit prédire la classe 0"


# ==============================================================================
# Validation stricte de la règle déterministe cachée (Employed = 1 <=> n_keys >= 3)
# ==============================================================================

@pytest.mark.parametrize("key_subset, expected_class", [
    ([], 0),
    (["Java"], 0),
    (["Node.js", "TypeScript"], 0),
    (["Java", "C#", "SQLite"], 1),
    (["Bash/Shell", "Node.js", "TypeScript", "MongoDB"], 1),
    (["Microsoft SQL Server", "npm", "jQuery", "TypeScript", "MongoDB", "SQLite"], 1),
])
def test_deterministic_rule_key_combinations(loaded_artifacts, sample_candidate_dict: dict,
                                              key_subset: list[str], expected_class: int):
    """Vérifie que le modèle respecte fidèlement la règle déterministe selon le nombre de clés."""
    model, _ = loaded_artifacts
    cand = sample_candidate_dict.copy()
    # Ajoute des technologies neutres pour s'assurer que seules les clés comptent
    all_techs = ["Python", "Git", "Docker"] + key_subset
    cand["HaveWorkedWith"] = ";".join(all_techs)
    cand["ComputerSkills"] = len(all_techs)

    df = pd.DataFrame([cand])
    pred = int(model.predict(df)[0])
    assert pred == expected_class, (
        f"Pour {len(key_subset)} technologies clés ({key_subset}), attendu {expected_class}, obtenu {pred}"
    )


def test_demographic_invariance_under_fixed_technology_rule(loaded_artifacts, key_technologies: list[str]):
    """Vérifie que les variations sociodémographiques (genre, âge, pays) ne perturbent pas la règle déterministe."""
    model, _ = loaded_artifacts

    # Profil 1 : 4 technologies clés -> doit toujours être 1
    techs_4keys = ";".join(key_technologies[:4] + ["Python", "Git"])
    # Profil 2 : 1 technologie clé -> doit toujours être 0
    techs_1key = ";".join(key_technologies[:1] + ["Python", "Docker"])

    variations = [
        {"Age": "<35", "Gender": "Woman", "Country": "France", "Accessibility": "No"},
        {"Age": ">35", "Gender": "Man", "Country": "Germany", "Accessibility": "Yes"},
        {"Age": "<35", "Gender": "NonBinary", "Country": "India", "Accessibility": "No"},
        {"Age": ">35", "Gender": "Woman", "Country": "United States of America", "Accessibility": "No"},
    ]

    for var in variations:
        cand_high = {
            "Age": var["Age"], "Accessibility": var["Accessibility"], "EdLevel": "Master",
            "Employment": 1, "Gender": var["Gender"], "MentalHealth": "No", "MainBranch": "Dev",
            "YearsCode": 10, "YearsCodePro": 5, "Country": var["Country"], "PreviousSalary": 60000.0,
            "HaveWorkedWith": techs_4keys, "ComputerSkills": 6
        }
        pred_high = int(model.predict(pd.DataFrame([cand_high]))[0])
        assert pred_high == 1, f"Échec prédiction 1 pour variation {var}"

        cand_low = cand_high.copy()
        cand_low["HaveWorkedWith"] = techs_1key
        cand_low["ComputerSkills"] = 3
        pred_low = int(model.predict(pd.DataFrame([cand_low]))[0])
        assert pred_low == 0, f"Échec prédiction 0 pour variation {var}"


# ==============================================================================
# Cas limites et valeurs extrêmes (Boundary & Edge cases)
# ==============================================================================

def test_extreme_numerical_boundaries(loaded_artifacts, sample_candidate_dict: dict):
    """Vérifie la robustesse face à des valeurs numériques extrêmes."""
    model, _ = loaded_artifacts

    # Débutant absolu (0 ans code, 0 ans pro, 1 $ salaire)
    cand_min = sample_candidate_dict.copy()
    cand_min["YearsCode"] = 0
    cand_min["YearsCodePro"] = 0
    cand_min["PreviousSalary"] = 1.0
    cand_min["HaveWorkedWith"] = "Python"
    cand_min["ComputerSkills"] = 1
    pred_min = int(model.predict(pd.DataFrame([cand_min]))[0])
    assert pred_min in (0, 1)

    # Vétéran senior avec très haut salaire (50 ans code, 1 000 000 $ salaire)
    cand_max = sample_candidate_dict.copy()
    cand_max["YearsCode"] = 50
    cand_max["YearsCodePro"] = 45
    cand_max["PreviousSalary"] = 1_000_000.0
    cand_max["HaveWorkedWith"] = "Python;Git"
    cand_max["ComputerSkills"] = 2
    pred_max = int(model.predict(pd.DataFrame([cand_max]))[0])
    assert pred_max in (0, 1)


def test_empty_technologies_profile(loaded_artifacts, sample_candidate_dict: dict):
    """Vérifie qu'un candidat sans aucune technologie déclarée est prédit 0 (Not Employed)."""
    model, _ = loaded_artifacts
    cand = sample_candidate_dict.copy()
    cand["HaveWorkedWith"] = ""
    cand["ComputerSkills"] = 0

    pred = int(model.predict(pd.DataFrame([cand]))[0])
    assert pred == 0, "Un candidat sans technologie (0 clé) doit être classé 0"


def test_many_non_key_technologies(loaded_artifacts, sample_candidate_dict: dict):
    """Vérifie qu'un candidat maîtrisant beaucoup de technologies mais AUCUNE clé reste classé 0."""
    model, _ = loaded_artifacts
    non_key_techs = [
        "Python", "R", "C++", "C", "Go", "Rust", "Ruby", "PHP", "Kotlin", "Swift",
        "HTML/CSS", "Git", "Docker", "Kubernetes", "AWS", "Google Cloud Platform",
        "Linux", "PostgreSQL", "MySQL", "Redis"
    ]
    cand = sample_candidate_dict.copy()
    cand["HaveWorkedWith"] = ";".join(non_key_techs)
    cand["ComputerSkills"] = len(non_key_techs)

    pred = int(model.predict(pd.DataFrame([cand]))[0])
    assert pred == 0, "Aucune des 20 technologies n'est une clé : attendu 0"


# ==============================================================================
# Tests du module CLI d'inférence (src/cli.py)
# ==============================================================================

def test_cli_predict_single_structure(loaded_artifacts, sample_candidate_dict: dict):
    """Vérifie la structure et la logique du dictionnaire retourné par cli.predict_single."""
    model, _ = loaded_artifacts
    result = predict_single(model, sample_candidate_dict)

    expected_keys = {
        "prediction", "label", "hyperplane_distance", "estimated_probability",
        "key_technologies_count", "key_technologies_detected",
        "deterministic_rule_met", "explanation"
    }
    assert expected_keys.issubset(result.keys())
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["estimated_probability"] <= 1.0
    assert result["deterministic_rule_met"] == (result["key_technologies_count"] >= 3)


def test_cli_batch_csv_prediction(loaded_artifacts, sample_batch_df: pd.DataFrame, tmp_path: Path):
    """Vérifie le traitement par lot depuis un fichier CSV."""
    model, _ = loaded_artifacts
    in_csv = tmp_path / "test_input.csv"
    out_csv = tmp_path / "test_output.csv"
    sample_batch_df.to_csv(in_csv, index=False)

    df_loaded = pd.read_csv(in_csv)
    preds = model.predict(df_loaded)
    df_loaded["Prediction"] = preds
    df_loaded["Prediction_Label"] = ["Employed" if p == 1 else "Not Employed" for p in preds]
    df_loaded.to_csv(out_csv, index=False)

    assert out_csv.exists()
    df_result = pd.read_csv(out_csv)
    assert len(df_result) == len(sample_batch_df)
    assert "Prediction" in df_result.columns
    assert "Prediction_Label" in df_result.columns


# ==============================================================================
# Tests des fonctions de validation de candidat (src/candidate.py)
# ==============================================================================

def test_validate_candidate_detects_inconsistencies():
    """Vérifie la détection d'erreurs de saisie dans le profil candidat."""
    # Erreur : années pro > années code
    errs = candidate.validate_candidate(years_code=3, years_pro=5, salary=50000.0, technologies=["Python"])
    assert len(errs) == 1
    assert "pratique du code" in errs[0]

    # Erreur : salaire négatif
    errs = candidate.validate_candidate(years_code=5, years_pro=3, salary=-100.0, technologies=["Python"])
    assert len(errs) == 1
    assert "négatif" in errs[0]

    # Erreur : doublons dans les technologies
    errs = candidate.validate_candidate(years_code=5, years_pro=3, salary=50000.0, technologies=["Python", "Python"])
    assert len(errs) == 1
    assert "une seule fois" in errs[0]

    # Profil valide -> 0 erreur
    errs = candidate.validate_candidate(years_code=10, years_pro=6, salary=60000.0, technologies=["Python", "Git"])
    assert len(errs) == 0


def test_deterministic_rule_summary_and_margin():
    """Vérifie deterministic_rule_summary et decision_margin_text."""
    res_pos = candidate.deterministic_rule_summary(["Java", "C#", "SQLite", "Python"])
    assert res_pos["count"] == 3
    assert res_pos["rule_met"] is True
    assert set(res_pos["key_techs"]) == {"Java", "C#", "SQLite"}

    res_neg = candidate.deterministic_rule_summary(["Python", "Git"])
    assert res_neg["count"] == 0
    assert res_neg["rule_met"] is False

    margin_text_pos = candidate.decision_margin_text(1.25)
    assert "classe 1" in margin_text_pos
    assert "+1.25" in margin_text_pos
    assert "Ce n'est pas une probabilité" in margin_text_pos

    margin_text_neg = candidate.decision_margin_text(-0.85)
    assert "classe 0" in margin_text_neg
    assert "-0.85" in margin_text_neg
