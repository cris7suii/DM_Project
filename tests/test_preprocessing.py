"""Tests unitaires pour le pipeline de prétraitement et d'encodage (src/features.py)."""
from __future__ import annotations

import pickle
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import features


# ==============================================================================
# Tests des fonctions utilitaires : split_techs & _log1p
# ==============================================================================

@pytest.mark.parametrize("input_text, expected", [
    ("Python;Git;TypeScript", ["Python", "Git", "TypeScript"]),
    ("  C# ;  Java  ; SQLite  ", ["C#", "Java", "SQLite"]),
    ("SingleTech", ["SingleTech"]),
    ("; ; Node.js ; ; ;", ["Node.js"]),
    ("", []),
    ("   ", []),
    (None, []),
    (12345, []),
])
def test_split_techs_parsing(input_text, expected):
    """Vérifie le découpage, le nettoyage des espaces et la robustesse de split_techs."""
    assert features.split_techs(input_text) == expected


def test_split_techs_is_picklable():
    """Vérifie que split_techs est sérialisable (requis par CountVectorizer dans scikit-learn)."""
    pickled = pickle.dumps(features.split_techs)
    unpickled = pickle.loads(pickled)
    assert callable(unpickled)
    assert unpickled("Python;C++") == ["Python", "C++"]


def test_log1p_math():
    """Vérifie la cohérence mathématique de _log1p sur scalaires et vecteurs."""
    assert features._log1p(0) == 0.0
    assert features._log1p(np.e - 1) == pytest.approx(1.0)
    arr = np.array([0.0, 10.0, 100.0])
    expected = np.log1p(arr)
    np.testing.assert_allclose(features._log1p(arr), expected)


# ==============================================================================
# Tests de construction du ColumnTransformer (build_preprocessor)
# ==============================================================================

def test_build_preprocessor_default_structure():
    """Vérifie la structure standard du ColumnTransformer."""
    prep = features.build_preprocessor(top_k_techs=40, nonneg=False)
    assert isinstance(prep, ColumnTransformer)

    transformer_names = [name for name, _, _ in prep.transformers]
    assert "num" in transformer_names
    assert "salary" in transformer_names
    assert "cat" in transformer_names
    assert "tech" in transformer_names


def test_build_preprocessor_nonneg_uses_minmax():
    """Vérifie que nonneg=True utilise MinMaxScaler (requis pour MultinomialNB)."""
    prep = features.build_preprocessor(top_k_techs=40, nonneg=True)
    trans_map = {name: trans for name, trans, _ in prep.transformers}
    num_pipeline = trans_map["num"]
    scaler = num_pipeline.named_steps["sc"]
    assert isinstance(scaler, MinMaxScaler)

    salary_pipeline = trans_map["salary"]
    assert isinstance(salary_pipeline.named_steps["sc"], MinMaxScaler)


def test_build_preprocessor_standard_uses_standard_scaler():
    """Vérifie que nonneg=False utilise StandardScaler par défaut."""
    prep = features.build_preprocessor(top_k_techs=40, nonneg=False)
    trans_map = {name: trans for name, trans, _ in prep.transformers}
    num_pipeline = trans_map["num"]
    scaler = num_pipeline.named_steps["sc"]
    assert isinstance(scaler, StandardScaler)


def test_build_preprocessor_zero_techs_excludes_vectorizer():
    """Vérifie que top_k_techs=0 exclut le transformer textuel."""
    prep = features.build_preprocessor(top_k_techs=0)
    transformer_names = [name for name, _, _ in prep.transformers]
    assert "tech" not in transformer_names


def test_build_preprocessor_drop_columns():
    """Vérifie l'exclusion ciblée de colonnes (pour ablations)."""
    prep = features.build_preprocessor(
        drop_cols=("ComputerSkills", "PreviousSalary", "HaveWorkedWith", "Gender")
    )
    trans_map = {name: trans for name, trans, _ in prep.transformers}
    cols_map = {name: cols for name, _, cols in prep.transformers}

    # PreviousSalary exclu -> pas de transformer salary
    assert "salary" not in trans_map

    # HaveWorkedWith exclu -> pas de transformer tech
    assert "tech" not in trans_map

    # ComputerSkills retiré de num
    assert "ComputerSkills" not in cols_map["num"]

    # Gender retiré de cat
    assert "Gender" not in cols_map["cat"]


# ==============================================================================
# Tests d'ajustement (fit) et de transformation sur données
# ==============================================================================

def test_preprocessor_fit_transform_sample(sample_batch_df: pd.DataFrame):
    """Vérifie l'ajustement et la transformation sur un jeu d'échantillons."""
    prep = features.build_preprocessor(top_k_techs=10)
    X = sample_batch_df.copy()
    X_trans = prep.fit_transform(X)

    # Doit avoir 4 lignes et un nombre strictement positif de colonnes
    assert X_trans.shape[0] == len(sample_batch_df)
    assert X_trans.shape[1] > 0

    # Aucun NaN ou inf dans la matrice résultante
    if hasattr(X_trans, "toarray"):
        dense = X_trans.toarray()
    else:
        dense = np.asarray(X_trans)
    assert not np.isnan(dense).any(), "La matrice transformée ne doit contenir aucun NaN"
    assert not np.isinf(dense).any(), "La matrice transformée ne doit contenir aucun inf"


def test_preprocessor_handles_missing_values(sample_batch_df: pd.DataFrame):
    """Vérifie la robustesse face aux valeurs manquantes (imputation médiane et mode)."""
    prep = features.build_preprocessor(top_k_techs=10)
    prep.fit(sample_batch_df)

    # Candidat avec plusieurs valeurs manquantes (NaN)
    corrupted_candidate = pd.DataFrame([{
        "Age": np.nan,
        "Accessibility": np.nan,
        "EdLevel": np.nan,
        "Employment": np.nan,
        "Gender": np.nan,
        "MentalHealth": np.nan,
        "MainBranch": np.nan,
        "YearsCode": np.nan,
        "YearsCodePro": np.nan,
        "Country": np.nan,
        "PreviousSalary": np.nan,
        "HaveWorkedWith": np.nan,
        "ComputerSkills": np.nan
    }])

    trans = prep.transform(corrupted_candidate)
    if hasattr(trans, "toarray"):
        trans_dense = trans.toarray()
    else:
        trans_dense = np.asarray(trans)

    assert not np.isnan(trans_dense).any(), "L'imputation doit éliminer tous les NaN"


def test_preprocessor_handles_unknown_categorical_values(sample_batch_df: pd.DataFrame):
    """Vérifie que les catégories non vues lors du fit (nouveau pays, etc.) ne lèvent pas d'erreur."""
    prep = features.build_preprocessor(top_k_techs=10)
    prep.fit(sample_batch_df)

    unseen_candidate = pd.DataFrame([{
        "Age": "UnknownAgeCategory",
        "Accessibility": "UnknownAccessibility",
        "EdLevel": "UnknownEdLevel",
        "Employment": 1,
        "Gender": "UnknownGender",
        "MentalHealth": "No",
        "MainBranch": "Dev",
        "YearsCode": 5,
        "YearsCodePro": 3,
        "Country": "UnknownCountryFictional",
        "PreviousSalary": 50000.0,
        "HaveWorkedWith": "UnknownTech1;UnknownTech2",
        "ComputerSkills": 2
    }])

    trans = prep.transform(unseen_candidate)
    assert trans.shape[0] == 1


def test_preprocessor_does_not_mutate_input_dataframe(sample_candidate_df: pd.DataFrame):
    """Vérifie que le préprocesseur ne modifie pas le DataFrame d'origine en place."""
    prep = features.build_preprocessor(top_k_techs=10)
    prep.fit(sample_candidate_df)

    original_dict = sample_candidate_df.to_dict(orient="records")
    _ = prep.transform(sample_candidate_df)
    after_dict = sample_candidate_df.to_dict(orient="records")

    assert original_dict == after_dict, "Le DataFrame d'entrée a été altéré pendant transform"
