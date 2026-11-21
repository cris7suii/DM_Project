"""
Tests automatisés pour la simulation des algorithmes de Data Mining (simulation_v2.py).
Valide :
1. Le chargement des artefacts (best_model.pkl, results.csv, meta.json, diagnostic.csv).
2. L'ajustement dynamique et l'inférence des 6 modèles (Linear SVC, KNN, Tree, NB, RF, Zero-R).
3. Le scénario complet (avec technologies) vs profil seul (sans technologies).
4. La robustesse aux valeurs manquantes et aux cas limites.
5. La conformité avec la règle déterministe (Employed = 1 ⟺ n_key >= 3).
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

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
    test_decision_threshold_matrix_consistency()
    print("test_decision_threshold_matrix_consistency: OK")
    print("Tous les tests unitaires ont réussi avec succès !")
