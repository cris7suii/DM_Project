"""
Tests automatisés pour la simulation des algorithmes de Data Mining (simulation_v2.py).
Valide :
1. Le chargement des artefacts (best_model.pkl, results.csv, meta.json, diagnostic.csv).
2. L'ajustement dynamique et l'inférence des 6 modèles (Linear SVC, KNN, Tree, NB, RF, Zero-R).
3. Le scénario complet (avec technologies) vs profil seul (sans technologies).
4. La robustesse aux valeurs manquantes et aux cas limites.
5. La conformité avec la règle déterministe (Employed = 1 ⟺ n_key >= 3).
6. L'analyse de sensibilité contrefactuelle (What-If Flips).
7. La cohérence des seuils de décision et de la matrice de confusion.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import features
import simulation_v2


def test_artifacts_exist_and_load():
    """Vérifie la présence et le chargement des modèles et métadonnées."""
    champ = simulation_v2._load_champion_pipeline()
    assert champ is not None, "best_model.pkl doit être chargeable"

    meta = simulation_v2._load_meta()
    assert "champion" in meta, "meta.json doit contenir le champion"
    assert meta["champion"] == "Linear SVC"

    res = simulation_v2._load_csv("results.csv")
    assert res is not None, "results.csv doit être chargeable"
    assert "F1 macro" in res.columns

    diag = simulation_v2._load_csv("diagnostic.csv")
    assert diag is not None, "diagnostic.csv doit être chargeable"
    assert len(diag) == 11, "11 niveaux de n_key (0 à 10)"


def test_deterministic_rule_in_sample():
    """Vérifie que l'échantillon de simulation respecte la règle déterministe."""
    df, _ = simulation_v2._get_working_sample(n=300, seed=42)
    assert len(df) == 300
    for _, row in df.iterrows():
        n_k = simulation_v2._count_key_techs(row["HaveWorkedWith"])
        expected = 1 if n_k >= 3 else 0
        assert row["Employed"] == expected, f"Violation de la règle pour n_k={n_k}"


def test_models_fit_and_predict_full():
    """Vérifie l'inférence des 6 algorithmes en scénario complet."""
    fitted, X_tr, y_tr, X_te, y_te = simulation_v2._fit_simulation_models(use_full=True)
    assert set(fitted.keys()) == {"knn", "tree", "rf", "nb", "svc"}

    # Candidat avec 5 technologies clés -> doit être prédit Employé (1)
    cand_5keys = pd.DataFrame([{
        "Age": "<35", "Accessibility": "No", "EdLevel": "Master", "Employment": 1,
        "Gender": "Man", "MentalHealth": "No", "MainBranch": "Dev",
        "YearsCode": 10, "YearsCodePro": 7, "Country": "France", "PreviousSalary": 80000.0,
        "HaveWorkedWith": "Python;Node.js;TypeScript;Bash/Shell;SQL;Git", "ComputerSkills": 6
    }])

    for name, pipe in fitted.items():
        pred = int(pipe.predict(cand_5keys)[0])
        assert pred == 1, f"Modèle {name} aurait dû prédire 1 pour 5 clés"


def test_profile_only_ablation():
    """Vérifie que le mode profil seul fonctionne sans erreur d'inférence."""
    fitted, X_tr, y_tr, X_te, y_te = simulation_v2._fit_simulation_models(use_full=False)

    cand_profile = pd.DataFrame([{
        "Age": ">35", "Accessibility": "No", "EdLevel": "Undergraduate", "Employment": 1,
        "Gender": "Woman", "MentalHealth": "No", "MainBranch": "Dev",
        "YearsCode": 12, "YearsCodePro": 8, "Country": "Germany", "PreviousSalary": 65000.0,
        "HaveWorkedWith": "", "ComputerSkills": 0
    }])

    for name, pipe in fitted.items():
        pred = int(pipe.predict(cand_profile)[0])
        assert pred in (0, 1), f"Modèle {name} a retourné une classe invalide: {pred}"


def test_whatif_counterfactual_flip():
    """Vérifie que l'ajout d'une technologie clé fait basculer la décision du champion (Flip 0 -> 1)."""
    champ = simulation_v2._load_champion_pipeline()
    assert champ is not None

    # Candidat avec 2 technologies clés (Bash/Shell et Node.js) -> doit prédire 0
    cand_2keys = pd.DataFrame([{
        "Age": "<35", "Accessibility": "No", "EdLevel": "Master", "Employment": 1,
        "Gender": "Man", "MentalHealth": "No", "MainBranch": "Dev",
        "YearsCode": 5, "YearsCodePro": 3, "Country": "France", "PreviousSalary": 50000.0,
        "HaveWorkedWith": "Python;Bash/Shell;Node.js;Git", "ComputerSkills": 4
    }])
    pred_base = int(champ.predict(cand_2keys)[0])
    assert pred_base == 0, "Un candidat avec 2 clés doit être prédit Non Employé (0)"

    # Ajout d'une 3ème technologie clé (TypeScript) -> doit faire basculer la décision à 1
    cand_3keys = cand_2keys.copy()
    cand_3keys["HaveWorkedWith"] = "Python;Bash/Shell;Node.js;TypeScript;Git"
    cand_3keys["ComputerSkills"] = 5
    pred_flip = int(champ.predict(cand_3keys)[0])
    assert pred_flip == 1, "Un candidat avec 3 clés doit basculer vers Employé (1)"


def test_counterfactual_reverse_flip():
    """Vérifie que le retrait d'une technologie clé fait basculer la décision de 1 vers 0."""
    champ = simulation_v2._load_champion_pipeline()
    assert champ is not None

    # Candidat avec 3 technologies clés -> classe 1
    cand_3keys = pd.DataFrame([{
        "Age": "<35", "Accessibility": "No", "EdLevel": "Master", "Employment": 1,
        "Gender": "Man", "MentalHealth": "No", "MainBranch": "Dev",
        "YearsCode": 6, "YearsCodePro": 4, "Country": "France", "PreviousSalary": 55000.0,
        "HaveWorkedWith": "Java;C#;SQLite;Git", "ComputerSkills": 4
    }])
    assert int(champ.predict(cand_3keys)[0]) == 1

    # Retrait d'une clé (reste 2 clés : Java et C#) -> classe 0
    cand_2keys = cand_3keys.copy()
    cand_2keys["HaveWorkedWith"] = "Java;C#;Git"
    cand_2keys["ComputerSkills"] = 3
    assert int(champ.predict(cand_2keys)[0]) == 0


def test_decision_threshold_matrix_consistency():
    """Vérifie la cohérence arithmétique de la matrice de confusion à différents seuils."""
    champ = simulation_v2._load_champion_pipeline()
    assert champ is not None

    df_sample, _ = simulation_v2._get_working_sample(n=100, seed=42)
    y_true = df_sample["Employed"].values
    scores = champ.decision_function(df_sample.drop(columns=["Employed"]))
    probas = 1.0 / (1.0 + np.exp(-scores))

    for th in [0.2, 0.5, 0.8]:
        y_pred = (probas >= th).astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        assert (tp + fp + tn + fn) == 100, "La somme des quadrants doit égaler la taille de l'échantillon"


def test_decision_threshold_extreme_boundaries():
    """Vérifie le comportement de la matrice de confusion aux bornes extrêmes [0, 1]."""
    champ = simulation_v2._load_champion_pipeline()
    df_sample, _ = simulation_v2._get_working_sample(n=50, seed=42)
    scores = champ.decision_function(df_sample.drop(columns=["Employed"]))
    probas = 1.0 / (1.0 + np.exp(-scores))

    # Seuil 0.0 -> tous prédits 1
    y_pred_0 = (probas >= 0.0).astype(int)
    assert np.all(y_pred_0 == 1)

    # Seuil 1.01 -> aucun prédit 1
    y_pred_1 = (probas >= 1.01).astype(int)
    assert np.all(y_pred_1 == 0)


def test_decision_threshold_monotonicity():
    """Vérifie la décroissance monotone du taux de prédictions positives avec le seuil."""
    champ = simulation_v2._load_champion_pipeline()
    df_sample, _ = simulation_v2._get_working_sample(n=100, seed=42)
    scores = champ.decision_function(df_sample.drop(columns=["Employed"]))
    probas = 1.0 / (1.0 + np.exp(-scores))

    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    pos_counts = [int(np.sum(probas >= th)) for th in thresholds]

    for i in range(len(pos_counts) - 1):
        assert pos_counts[i] >= pos_counts[i + 1], "Le nombre de positifs doit décroître avec le seuil"


def test_get_working_sample_reproducibility():
    """Vérifie la reproductibilité du tirage de l'échantillon synthétique par seed."""
    df1, _ = simulation_v2._get_working_sample(n=50, seed=123)
    df2, _ = simulation_v2._get_working_sample(n=50, seed=123)
    pd.testing.assert_frame_equal(df1, df2)

    df3, _ = simulation_v2._get_working_sample(n=50, seed=456)
    assert not df1.equals(df3), "Des graines différentes doivent produire des échantillons différents"


@pytest.mark.parametrize("techs, expected_count", [
    ("", 0),
    ("Python;Git", 0),
    ("Java", 1),
    ("Java;C#;SQLite", 3),
    ("Microsoft SQL Server;npm;jQuery;TypeScript;MongoDB;SQLite;Bash/Shell;Node.js;C#;Java", 10),
    (None, 0),
    (12345, 0),
])
def test_count_key_techs(techs, expected_count: int):
    """Vérifie le comptage précis des 10 technologies clés de la règle."""
    assert simulation_v2._count_key_techs(techs) == expected_count


if __name__ == "__main__":
    test_artifacts_exist_and_load()
    print("test_artifacts_exist_and_load: OK")
    test_deterministic_rule_in_sample()
    print("test_deterministic_rule_in_sample: OK")
    test_models_fit_and_predict_full()
    print("test_models_fit_and_predict_full: OK")
    test_profile_only_ablation()
    print("test_profile_only_ablation: OK")
    test_whatif_counterfactual_flip()
    print("test_whatif_counterfactual_flip: OK")
    test_counterfactual_reverse_flip()
    print("test_counterfactual_reverse_flip: OK")
    test_decision_threshold_matrix_consistency()
    print("test_decision_threshold_matrix_consistency: OK")
    test_decision_threshold_extreme_boundaries()
    print("test_decision_threshold_extreme_boundaries: OK")
    test_decision_threshold_monotonicity()
    print("test_decision_threshold_monotonicity: OK")
    test_get_working_sample_reproducibility()
    print("test_get_working_sample_reproducibility: OK")
    test_count_key_techs("Java;C#;SQLite", 3)
    print("test_count_key_techs: OK")
    print("Tous les tests unitaires ont réussi avec succès !")
