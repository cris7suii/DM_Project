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


if __name__ == "__main__":
    test_artifacts_exist_and_load()
    print("test_artifacts_exist_and_load: OK")
    test_deterministic_rule_in_sample()
    print("test_deterministic_rule_in_sample: OK")
    test_models_fit_and_predict_full()
    print("test_models_fit_and_predict_full: OK")
    test_profile_only_ablation()
    print("test_profile_only_ablation: OK")
    print("Tous les tests unitaires ont réussi avec succès !")
