"""Prétraitement partagé entre les notebooks et l'application Streamlit.

Le ColumnTransformer est construit ici (non ajusté). Il est ensuite placé DANS un
Pipeline avec le modèle, de sorte que GridSearchCV le ré-ajuste à chaque pli de
validation croisée (aucune fuite de données).
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler,
                                   OneHotEncoder, StandardScaler)

TARGET = "Employed"

NUM_COLS = ["YearsCode", "YearsCodePro", "PreviousSalary", "ComputerSkills", "Employment"]
CAT_COLS = ["Age", "Accessibility", "EdLevel", "Gender", "MentalHealth", "MainBranch", "Country"]
TECH_COL = "HaveWorkedWith"

# Attributs sensibles (analysés dans la section « équité » du notebook 03)
SENSITIVE_COLS = ["Age", "Gender", "MentalHealth", "Accessibility"]


def split_techs(text) -> list[str]:
    """'C++;Python;Git' -> ['C++', 'Python', 'Git'] (fonction de niveau module => picklable)."""
    if not isinstance(text, str) or not text:
        return []
    return [t.strip() for t in text.split(";") if t.strip()]


def _log1p(x):
    return np.log1p(x)


def build_preprocessor(top_k_techs: int = 40, nonneg: bool = False,
                       drop_cols: tuple[str, ...] = ()) -> ColumnTransformer:
    """Construit le ColumnTransformer.

    top_k_techs : nombre de technologies les plus fréquentes encodées en multi-hot (0 = aucune)
    nonneg      : True -> MinMaxScaler (requis par MultinomialNB, pas de valeurs négatives)
    drop_cols   : colonnes à exclure (ablations : 'ComputerSkills', variables sensibles, ...)
    """
    scaler = MinMaxScaler if nonneg else StandardScaler
    num_cols = [c for c in NUM_COLS if c not in drop_cols and c != "PreviousSalary"]
    cat_cols = [c for c in CAT_COLS if c not in drop_cols]

    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", scaler()),
        ]), num_cols))
    if "PreviousSalary" not in drop_cols:
        # salaire très asymétrique -> log1p avant mise à l'échelle
        transformers.append(("salary", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(_log1p, feature_names_out="one-to-one")),
            ("sc", scaler()),
        ]), ["PreviousSalary"]))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            # les pays rares sont regroupés dans une catégorie « infrequent »
            ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100)),
        ]), cat_cols))
    if top_k_techs and TECH_COL not in drop_cols:
        transformers.append(("tech", CountVectorizer(
            tokenizer=split_techs, token_pattern=None, lowercase=False,
            binary=True, max_features=top_k_techs), TECH_COL))
    return ColumnTransformer(transformers, sparse_threshold=1.0)
