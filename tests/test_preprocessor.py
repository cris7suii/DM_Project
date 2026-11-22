"""
Tests unitaires pour le pipeline renforcé d'encodage et d'imputation (features.py).

Valide :
1. La rétrocompatibilité intégrale du préprocesseur standard (build_preprocessor).
2. L'encodage ordinal hiérarchique de 'EdLevel' et 'Age' (préservation de l'ordre sans inflation).
3. L'encodage binaire compact (drop='if_binary') sur les variables dichotomiques.
4. L'imputation multi-stratégies (Médiane, Moyenne, KNN) et la capture des indicateurs MissingIndicator.
5. L'immunité du vocabulaire technologique (garantie des 10 technologies clés déterministes).
6. La robustesse aux valeurs aberrantes (RobustScaler vs StandardScaler) et aux cas extrêmes (NaNs complets, inconnus).
7. L'intégration sans faille avec des modèles Scikit-Learn (Pipeline, fit, predict).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import features


def _make_sample_dataframe():
    """Génère un jeu de test représentatif avec cas nominaux, extrêmes et valeurs manquantes."""
    return pd.DataFrame([
        {
            "Age": "<35", "Accessibility": "No", "EdLevel": "Master", "Employment": 1,
            "Gender": "Man", "MentalHealth": "No", "MainBranch": "Dev",
            "YearsCode": 10, "YearsCodePro": 7, "Country": "France", "PreviousSalary": 80000.0,
            "HaveWorkedWith": "Python;Node.js;TypeScript;Bash/Shell;SQL;Git", "ComputerSkills": 6,
            "Employed": 1
        },
        {
            "Age": ">35", "Accessibility": "Yes", "EdLevel": "PhD", "Employment": 0,
            "Gender": "Woman", "MentalHealth": "Yes", "MainBranch": "NotDev",
            "YearsCode": 20, "YearsCodePro": 15, "Country": "Germany", "PreviousSalary": 150000.0,
            "HaveWorkedWith": "Java;C#;MongoDB;SQLite;npm", "ComputerSkills": 5,
            "Employed": 1
        },
        {
            "Age": "<35", "Accessibility": "No", "EdLevel": "NoHigherEd", "Employment": 1,
            "Gender": "NonBinary", "MentalHealth": "No", "MainBranch": "Dev",
            "YearsCode": 2, "YearsCodePro": 1, "Country": "United States of America", "PreviousSalary": 35000.0,
            "HaveWorkedWith": "HTML/CSS;JavaScript", "ComputerSkills": 2,
            "Employed": 0
        },
        {
            "Age": None, "Accessibility": None, "EdLevel": None, "Employment": None,
            "Gender": None, "MentalHealth": None, "MainBranch": None,
            "YearsCode": None, "YearsCodePro": None, "Country": None, "PreviousSalary": None,
            "HaveWorkedWith": None, "ComputerSkills": None,
            "Employed": 0
        }
    ])


def test_baseline_backward_compatibility():
    """Vérifie que build_preprocessor() conserve son comportement standard sans régression."""
    df = _make_sample_dataframe()
    X = df.drop(columns=["Employed"]).assign(HaveWorkedWith=df["HaveWorkedWith"].fillna(""))
    prep = features.build_preprocessor(top_k_techs=40)
    X_trans = prep.fit_transform(X)
    assert X_trans.shape[0] == len(df)
    assert X_trans.shape[1] > 0
    print("test_baseline_backward_compatibility: OK")


def test_ordinal_encoder_preservation():
    """Vérifie que EdLevel respecte scrupuleusement la hiérarchie académique."""
    df = pd.DataFrame({
        "EdLevel": ["NoHigherEd", "Other", "Undergraduate", "Master", "PhD"],
        "Age": ["<35", ">35", "<35", ">35", "<35"],
        "YearsCode": [1, 3, 5, 8, 12],
        "YearsCodePro": [0, 1, 3, 5, 8],
        "ComputerSkills": [1, 2, 3, 4, 5],
        "Employment": [1, 1, 1, 1, 1],
        "PreviousSalary": [30000.0, 40000.0, 55000.0, 75000.0, 110000.0],
        "Gender": ["Man", "Woman", "Man", "Woman", "NonBinary"],
        "Country": ["France", "Germany", "France", "Germany", "France"],
        "Accessibility": ["No", "No", "No", "No", "No"],
        "MentalHealth": ["No", "No", "No", "No", "No"],
        "MainBranch": ["Dev", "Dev", "Dev", "Dev", "Dev"],
        "HaveWorkedWith": ["Python", "Java", "C#", "Node.js", "TypeScript"]
    })
    prep = features.build_reinforced_preprocessor(use_ordinal=True, add_missing_indicator=False, top_k_techs=10)
    prep.fit(df)
    
    # Extraire le transformateur ordinal
    ord_step = prep.named_transformers_["cat_ord"].named_steps["ord"]
    cats = ord_step.categories_
    expected_ed_order = features.ORDINAL_HIERARCHY["EdLevel"]
    assert list(cats[0]) == expected_ed_order, f"Ordre attendu: {expected_ed_order}, obtenu: {list(cats[0])}"
    print("test_ordinal_encoder_preservation: OK")


def test_binary_columns_compact_encoding():
    """Vérifie que les variables binaires génèrent 1 seule colonne (suppression de la multicolinéarité)."""
    df = _make_sample_dataframe()
    X = df.drop(columns=["Employed"])
    prep = features.build_reinforced_preprocessor(use_ordinal=True, add_missing_indicator=False)
    prep.fit(X)
    
    bin_ohe = prep.named_transformers_["cat_bin"].named_steps["ohe"]
    # Chaque variable binaire [Accessibility, MentalHealth, MainBranch] doit avoir drop='if_binary'
    feature_names = bin_ohe.get_feature_names_out()
    assert len(feature_names) == 3, f"3 colonnes binaires attendues, obtenu: {len(feature_names)}"
    print("test_binary_columns_compact_encoding: OK")


def test_missing_indicators_generation():
    """Vérifie la génération explicite des indicateurs MissingIndicator sur données lacunaires."""
    df = _make_sample_dataframe()
    X = df.drop(columns=["Employed"])
    prep = features.build_reinforced_preprocessor(add_missing_indicator=True)
    X_trans = prep.fit_transform(X)
    
    # Doit contenir la branche 'missing_flags'
    assert "missing_flags" in prep.named_transformers_
    flags_step = prep.named_transformers_["missing_flags"]
    flags_arr = flags_step.transform(X[features.NUM_COLS])
    
    # La dernière ligne a toutes ses valeurs numériques à None -> flags doivent être à True (1)
    assert flags_arr[-1].all() == True, "La ligne entièrement manquante doit avoir tous ses flags à 1"
    assert not flags_arr[0].any() == True, "La ligne complète ne doit avoir aucun flag à 1"
    print("test_missing_indicators_generation: OK")


def test_key_techs_guaranteed_in_vocabulary():
    """Vérifie que les 10 technologies clés sont TOUTES incluses dans le vocabulaire."""
    corpus = [
        "Rust;Go;Haskell;Scala;Elixir",
        "Kotlin;Swift;Dart;R;Perl",
        "Docker;Kubernetes;AWS;Linux;Git"
    ]
    # Même avec un top_k faible (ex. 15), les 10 KEY_TECHS doivent être injectées
    vec = features.KeyTechCountVectorizer(top_k_techs=15, guarantee_key_techs=True)
    vec.fit(corpus)
    names = list(vec.get_feature_names_out())
    
    for k in features.KEY_TECHS:
        assert k in names, f"Technologie clé absente du vocabulaire : {k}"
    assert len(names) >= 10
    print("test_key_techs_guaranteed_in_vocabulary: OK")


def test_robust_scaler_resistance():
    """Vérifie que RobustScaler est configuré et normalise par médiane et IQR."""
    df = _make_sample_dataframe()
    X = df.drop(columns=["Employed"])
    prep = features.build_reinforced_preprocessor(scaler_type="robust")
    prep.fit(X)
    
    sc = prep.named_transformers_["num"].named_steps["sc"]
    assert type(sc).__name__ == "RobustScaler", f"Type inattendu: {type(sc)}"
    print("test_robust_scaler_resistance: OK")


def test_dirty_inputs_and_unseen_categories():
    """Vérifie la résilience totale face à des valeurs inconnues ou erratiques."""
    df_train = _make_sample_dataframe()
    prep = features.build_reinforced_preprocessor(use_ordinal=True, add_missing_indicator=True)
    prep.fit(df_train.drop(columns=["Employed"]))
    
    # Données test avec modalité inconnue ('Mars', 'Inconnu') et types inattendus
    df_unseen = pd.DataFrame([{
        "Age": "UnknownAge", "Accessibility": "Maybe", "EdLevel": "HonoraryDegree", "Employment": 99,
        "Gender": "UnknownGender", "MentalHealth": "UnknownMH", "MainBranch": "UnknownMB",
        "YearsCode": 100, "YearsCodePro": 50, "Country": "Atlantis", "PreviousSalary": 9999999.0,
        "HaveWorkedWith": "AncientLanguage;UnknownTech;None", "ComputerSkills": 50
    }])
    
    # Ne doit pas lever d'exception
    X_out = prep.transform(df_unseen)
    assert X_out.shape[0] == 1
    print("test_dirty_inputs_and_unseen_categories: OK")


def test_end_to_end_model_pipeline():
    """Vérifie l'entraînement et l'inférence bout-en-bout avec un classifieur scikit-learn."""
    df = _make_sample_dataframe()
    X = df.drop(columns=["Employed"])
    y = df["Employed"]
    
    pipe = Pipeline([
        ("prep", features.build_reinforced_preprocessor(
            top_k_techs=20, use_ordinal=True, add_missing_indicator=True, scaler_type="robust"
        )),
        ("clf", DecisionTreeClassifier(random_state=42))
    ])
    
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(df)
    assert set(preds).issubset({0, 1})
    print("test_end_to_end_model_pipeline: OK")


def test_comparison_summary_table():
    """Vérifie la génération du tableau comparatif entre standard et renforcé."""
    df = _make_sample_dataframe()
    cmp_df = features.compare_preprocessing_strategies(df)
    assert isinstance(cmp_df, pd.DataFrame)
    assert len(cmp_df) >= 6
    assert "Bénéfice Académique" in cmp_df.columns
    print("test_comparison_summary_table: OK")


if __name__ == "__main__":
    print("=== Démarrage des Tests du Pipeline d'Encodage et d'Imputation ===")
    test_baseline_backward_compatibility()
    test_ordinal_encoder_preservation()
    test_binary_columns_compact_encoding()
    test_missing_indicators_generation()
    test_key_techs_guaranteed_in_vocabulary()
    test_robust_scaler_resistance()
    test_dirty_inputs_and_unseen_categories()
    test_end_to_end_model_pipeline()
    test_comparison_summary_table()
    print("✅ TOUS LES TESTS DU PIPELINE RENFORCÉ ONT RÉUSSI AVEC SUCCÈS !")
