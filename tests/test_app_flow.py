"""Tests unitaires de la logique applicative et des flux fonctionnels de src/app.py."""
from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pandas as pd
import pytest
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from artifacts import ArtifactLoadError


class MockStreamlitContainer:
    """Mock générique pour les contextes et conteneurs Streamlit (tabs, columns, form)."""
    def __init__(self, name: str = "container"):
        self.name = name

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def selectbox(self, *args, **kw):
        return st.selectbox(*args, **kw)

    def number_input(self, *args, **kw):
        return st.number_input(*args, **kw)

    def slider(self, *args, **kw):
        return st.slider(*args, **kw)

    def radio(self, *args, **kw):
        return st.radio(*args, **kw)

    def multiselect(self, *args, **kw):
        return st.multiselect(*args, **kw)

    def form_submit_button(self, *args, **kw):
        return st.form_submit_button(*args, **kw)

    def button(self, *args, **kw):
        return st.button(*args, **kw)

    def metric(self, *args, **kw):
        return st.metric(*args, **kw)

    def markdown(self, *args, **kw):
        return st.markdown(*args, **kw)

    def write(self, *args, **kw):
        return st.write(*args, **kw)

    def caption(self, *args, **kw):
        return st.caption(*args, **kw)

    def dataframe(self, *args, **kw):
        return st.dataframe(*args, **kw)

    def bar_chart(self, *args, **kw):
        return st.bar_chart(*args, **kw)

    def pyplot(self, *args, **kw):
        return st.pyplot(*args, **kw)

    def error(self, *args, **kw):
        return st.error(*args, **kw)

    def info(self, *args, **kw):
        return st.info(*args, **kw)

    def download_button(self, *args, **kw):
        return st.download_button(*args, **kw)


def _setup_app_mocks(monkeypatch, submit_form: bool = True, invalid_years: bool = False):
    """Configure un environnement de mocks complet pour l'exécution fonctionnelle d'app.py."""
    try:
        st.cache_resource.clear()
        st.cache_data.clear()
    except Exception:
        pass

    recorded_calls = {
        "errors": [],
        "infos": [],
        "downloads": [],
        "metrics": [],
        "stopped": False
    }

    def mock_columns(spec, *args, **kwargs):
        n = spec if isinstance(spec, int) else len(spec)
        return [MockStreamlitContainer(f"col_{i}") for i in range(n)]

    def mock_tabs(tab_names):
        return [MockStreamlitContainer(f"tab_{name}") for name in tab_names]

    def mock_number_input(label, min_v=0, max_v=100, default=0, *args, **kw):
        if invalid_years and "professionnelle" in label:
            return 20  # Force years_pro > years_code (incohérence)
        if "pratique du code" in label:
            return 8
        if "professionnelle" in label:
            return 5
        if "salaire" in label.lower():
            return 65000
        if "autres technologies" in label.lower():
            return 2
        return default

    def mock_multiselect(label, options, default=None, *args, **kw):
        return ["Python", "Git", "TypeScript", "Node.js", "MongoDB"]  # 3 clés -> Employed (1)

    def mock_download_button(label, data, file_name, mime=None):
        recorded_calls["downloads"].append({"label": label, "data": data, "file_name": file_name})

    def mock_stop():
        recorded_calls["stopped"] = True
        raise StopIteration("st.stop() simulation")

    monkeypatch.setattr(st, "set_page_config", lambda *args, **kw: None)
    monkeypatch.setattr(st, "tabs", mock_tabs)
    monkeypatch.setattr(st, "columns", mock_columns)
    monkeypatch.setattr(st, "form", lambda *args, **kw: MockStreamlitContainer("form"))
    monkeypatch.setattr(st, "form_submit_button", lambda *args, **kw: submit_form)
    monkeypatch.setattr(st, "selectbox", lambda label, options, *args, **kw: options[0] if options else None)
    monkeypatch.setattr(st, "number_input", mock_number_input)
    monkeypatch.setattr(st, "multiselect", mock_multiselect)
    monkeypatch.setattr(st, "download_button", mock_download_button)
    monkeypatch.setattr(st, "error", lambda msg: recorded_calls["errors"].append(msg))
    monkeypatch.setattr(st, "info", lambda msg: recorded_calls["infos"].append(msg))
    monkeypatch.setattr(st, "stop", mock_stop)
    monkeypatch.setattr(st, "markdown", lambda *args, **kw: None)
    monkeypatch.setattr(st, "write", lambda *args, **kw: None)
    monkeypatch.setattr(st, "caption", lambda *args, **kw: None)
    monkeypatch.setattr(st, "metric", lambda *args, **kw: None)
    monkeypatch.setattr(st, "dataframe", lambda *args, **kw: None)
    monkeypatch.setattr(st, "bar_chart", lambda *args, **kw: None)

    return recorded_calls


def test_app_nominal_inference_and_audit_export(monkeypatch):
    """Vérifie l'exécution complète d'app.py, l'inférence candidat et l'export du rapport d'audit JSON."""
    calls = _setup_app_mocks(monkeypatch, submit_form=True, invalid_years=False)

    # Exécution du script app.py
    if "app" in sys.modules:
        del sys.modules["app"]
    runpy.run_module("app", run_name="__main__")

    assert len(calls["errors"]) == 0, f"Erreurs inattendues levées : {calls['errors']}"
    assert len(calls["downloads"]) == 1, "Le bouton de téléchargement du rapport doit avoir été généré"

    # Vérification du contenu du rapport d'audit JSON exportable
    download_item = calls["downloads"][0]
    assert download_item["file_name"].startswith("audit_candidat_")
    report = json.loads(download_item["data"])

    assert report["system"] == "EmployML Data Mining Governance Platform"
    assert "candidate_profile" in report
    assert "audit_verdict" in report
    assert report["audit_verdict"]["classification"] == 1
    assert report["audit_verdict"]["deterministic_rule_met"] is True
    assert report["audit_verdict"]["key_techs_count"] >= 3


def test_app_validation_error_stops_execution(monkeypatch):
    """Vérifie que la saisie d'un candidat invalide (années pro > années code) affiche une erreur et stoppe."""
    calls = _setup_app_mocks(monkeypatch, submit_form=True, invalid_years=True)

    if "app" in sys.modules:
        del sys.modules["app"]
    with pytest.raises(StopIteration, match="st.stop"):
        runpy.run_module("app", run_name="__main__")

    assert len(calls["errors"]) >= 1
    assert any("pratique du code" in err for err in calls["errors"])
    assert calls["stopped"] is True


def test_app_startup_artifact_load_error_stops_execution(monkeypatch):
    """Vérifie qu'un échec de chargement des modèles au démarrage affiche st.error et st.stop."""
    calls = _setup_app_mocks(monkeypatch, submit_form=False)

    import artifacts
    monkeypatch.setattr(artifacts, "load_model_and_meta", MagicMock(side_effect=ArtifactLoadError("Modèle absent")))

    if "app" in sys.modules:
        del sys.modules["app"]
    with pytest.raises(StopIteration, match="st.stop"):
        runpy.run_module("app", run_name="__main__")

    assert any("Impossible de charger les artefacts" in err for err in calls["errors"])
    assert calls["stopped"] is True
