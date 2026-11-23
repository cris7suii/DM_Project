"""Tests des onglets AdaBoost / XGBoost de la simulation et du réglage interactif (GridSearchCV)."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import features  # noqa: E402
import simulation_v2  # noqa: E402

xgboost = pytest.importorskip("xgboost", reason="xgboost non installé")


@pytest.fixture
def synthetic_split(monkeypatch):
    """Échantillon synthétique (règle exacte) : indépendant de data/processed/ (non versionné)."""
    monkeypatch.setattr(simulation_v2, "_find_real_dataset", lambda: None)
    simulation_v2._get_working_sample.clear()
    simulation_v2._fit_simulation_models.clear()
    return simulation_v2._fit_simulation_models(use_full=True)


@pytest.fixture
def candidate_df() -> pd.DataFrame:
    return pd.DataFrame([{
        "Age": "<35", "Accessibility": "No", "EdLevel": "Master", "Employment": 1,
        "Gender": "Man", "MentalHealth": "No", "MainBranch": "Dev",
        "YearsCode": 10, "YearsCodePro": 7, "Country": "France", "PreviousSalary": 80000.0,
        "HaveWorkedWith": "Node.js;TypeScript;Bash/Shell;Microsoft SQL Server;Java;Git", "ComputerSkills": 6,
    }])


def test_preprocessor_pipeline_is_picklable(synthetic_split):
    """Un Pipeline ajusté doit pouvoir être sérialisé (joblib/pickle) : requis pour models/best_model.pkl."""
    fitted, *_ = synthetic_split
    restored = pickle.loads(pickle.dumps(fitted["svc"]))
    assert hasattr(restored, "predict")


def test_boosting_models_structure(synthetic_split):
    fitted, _, _, X_te, _ = synthetic_split

    ada = fitted["ada"].named_steps["clf"]
    assert len(ada.estimators_) >= 1
    assert all(est.get_depth() == 1 for est in ada.estimators_), "les souches doivent avoir une profondeur 1"
    assert len(ada.estimator_weights_) >= len(ada.estimators_)
    assert len(ada.estimator_errors_) >= len(ada.estimators_)

    xgb = fitted["xgb"]
    proba = xgb.predict_proba(X_te.iloc[:5])
    assert proba.shape == (5, 2) and np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert len(xgb.named_steps["clf"].feature_importances_) > 0


def test_boosting_predict_employed_for_many_key_techs(synthetic_split, candidate_df):
    """Candidat avec 5 technologies clés : la règle déterministe doit être retrouvée par les deux modèles."""
    fitted, *_ = synthetic_split
    for name in ("ada", "xgb"):
        assert int(fitted[name].predict(candidate_df)[0]) == 1, f"{name} aurait dû prédire Employé"


def test_run_mini_search_returns_ranked_results(synthetic_split):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier

    _, X_tr, y_tr, _, _ = synthetic_split
    pre = features.build_preprocessor(top_k_techs=40, nonneg=False)
    clf = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=42), random_state=42)
    grid = {"clf__n_estimators": [10, 20], "clf__learning_rate": [0.5, 1.0]}
    best, res = simulation_v2._run_mini_search(pre, clf, grid, X_tr, y_tr)

    assert set(best) == {"n_estimators", "learning_rate"}
    assert len(res) == 4
    assert res["F1-macro CV"].is_monotonic_decreasing, "résultats triés du meilleur au moins bon"
    assert res.iloc[0]["F1-macro CV"] == res["F1-macro CV"].max()


def test_boosting_views_mocked_execution(synthetic_split, candidate_df, monkeypatch):
    """Exécute les deux vues Streamlit (mock complet de l'interface) pour les deux scénarios."""
    import streamlit as st

    class MockContext:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def mock_columns(spec, *args, **kwargs):
        n = spec if isinstance(spec, int) else len(spec)
        return [MockContext() for _ in range(n)]

    recorded = {"metrics": 0}
    monkeypatch.setattr(st, "columns", mock_columns)
    monkeypatch.setattr(st, "expander", lambda *a, **kw: MockContext())
    monkeypatch.setattr(st, "markdown", lambda *a, **kw: None)
    monkeypatch.setattr(st, "caption", lambda *a, **kw: None)
    monkeypatch.setattr(st, "write", lambda *a, **kw: None)
    monkeypatch.setattr(st, "json", lambda *a, **kw: None)
    monkeypatch.setattr(st, "metric", lambda *a, **kw: recorded.__setitem__("metrics", recorded["metrics"] + 1))
    monkeypatch.setattr(st, "dataframe", lambda *a, **kw: None)
    monkeypatch.setattr(st, "pyplot", lambda *a, **kw: None)
    monkeypatch.setattr(st, "slider", lambda label, min_v, max_v, default, *a, **kw: default)
    monkeypatch.setattr(st, "button", lambda *a, **kw: False)

    _, X_tr, y_tr, X_te, y_te = synthetic_split
    for use_full in (True, False):
        p_ada = simulation_v2._render_adaboost_view(candidate_df, use_full, X_tr, y_tr, X_te, y_te, 1)
        p_xgb = simulation_v2._render_xgboost_view(candidate_df, use_full, X_tr, y_tr, X_te, y_te, 1)
        assert p_ada in (0, 1) and p_xgb in (0, 1)
        if use_full:
            assert p_ada == 1 and p_xgb == 1
    assert recorded["metrics"] == 16  # 4 métriques × 2 vues × 2 scénarios
