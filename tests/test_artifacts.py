"""Tests de validation des artefacts de modèles et métadonnées du projet."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from artifacts import ArtifactLoadError, load_model_and_meta, load_result_csv


def _write_valid_artifacts(directory: Path) -> None:
    model = DummyClassifier(strategy="most_frequent").fit([[0], [1]], [0, 1])
    joblib.dump(model, directory / "best_model.pkl")
    (directory / "meta.json").write_text(
        json.dumps({"champion": "Dummy", "technologies": ["Python"], "countries": ["France"]}),
        encoding="utf-8",
    )


def test_load_model_and_meta_validates_expected_files(tmp_path):
    _write_valid_artifacts(tmp_path)
    model, meta = load_model_and_meta(tmp_path)
    assert hasattr(model, "predict")
    assert meta["champion"] == "Dummy"


def test_load_model_and_meta_reports_missing_artifacts(tmp_path):
    with pytest.raises(ArtifactLoadError, match="absent"):
        load_model_and_meta(tmp_path)


def test_load_model_and_meta_rejects_invalid_metadata(tmp_path):
    _write_valid_artifacts(tmp_path)
    (tmp_path / "meta.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ArtifactLoadError, match="champs requis"):
        load_model_and_meta(tmp_path)


def test_load_model_and_meta_rejects_non_dict_json(tmp_path):
    _write_valid_artifacts(tmp_path)
    (tmp_path / "meta.json").write_text("[\"not_a_dict\"]", encoding="utf-8")
    with pytest.raises(ArtifactLoadError, match="doit contenir un objet JSON"):
        load_model_and_meta(tmp_path)


def test_load_model_and_meta_rejects_corrupted_json(tmp_path):
    _write_valid_artifacts(tmp_path)
    (tmp_path / "meta.json").write_text("{malformed json", encoding="utf-8")
    with pytest.raises(ArtifactLoadError, match="illisible ou contient un JSON invalide"):
        load_model_and_meta(tmp_path)


def test_load_model_and_meta_rejects_non_list_fields(tmp_path):
    _write_valid_artifacts(tmp_path)
    (tmp_path / "meta.json").write_text(
        json.dumps({"champion": "Dummy", "technologies": "not-a-list", "countries": ["France"]}),
        encoding="utf-8",
    )
    with pytest.raises(ArtifactLoadError, match="invalides"):
        load_model_and_meta(tmp_path)


# ==============================================================================
# Validation des artefacts réels du projet dans models/
# ==============================================================================

EXPECTED_ARTIFACT_FILES = [
    "best_model.pkl",
    "preprocessor.pkl",
    "meta.json",
    "results.csv",
    "profile_only_results.csv",
    "ablation.csv",
    "diagnostic.csv",
    "fairness.csv",
]


@pytest.mark.parametrize("filename", EXPECTED_ARTIFACT_FILES)
def test_real_model_artifacts_exist(models_dir: Path, filename: str):
    """Vérifie la présence physique de chaque artefact requis dans models/."""
    file_path = models_dir / filename
    assert file_path.exists(), f"L'artefact requis {filename} est introuvable dans {models_dir}"
    assert file_path.stat().st_size > 0, f"L'artefact {filename} ne doit pas être vide"


def test_real_artifacts_load_with_helper(models_dir: Path):
    """Vérifie que load_model_and_meta charge avec succès les vrais artefacts de production."""
    model, meta = load_model_and_meta(models_dir)
    assert model is not None
    assert hasattr(model, "predict")
    assert meta["champion"] == "Linear SVC"


def test_real_meta_json_structure(models_dir: Path, key_technologies: list[str]):
    """Vérifie le contenu et les métriques clés de meta.json."""
    meta_path = models_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # Vérification des clés de premier niveau
    for key in ["champion", "best_params", "technologies", "countries", "n_train", "target_rate"]:
        assert key in meta, f"Clé '{key}' manquante dans meta.json"

    assert meta["champion"] == "Linear SVC"
    assert isinstance(meta["technologies"], list)
    assert len(meta["technologies"]) >= 10
    assert isinstance(meta["countries"], list)
    assert len(meta["countries"]) > 50
    assert meta["n_train"] > 0
    assert 0.0 < meta["target_rate"] < 1.0

    # Vérifie que les 10 technologies clés de la règle déterministe sont présentes
    for tech in key_technologies:
        assert tech in meta["technologies"], f"Technologie clé '{tech}' absente de meta['technologies']"


def test_real_champion_pipeline_structure(loaded_artifacts):
    """Vérifie la conformité de l'architecture du Pipeline champion scikit-learn."""
    model, meta = loaded_artifacts
    assert hasattr(model, "named_steps"), "Le champion doit être une Pipeline scikit-learn"
    assert "prep" in model.named_steps, "Le pipeline doit comporter l'étape 'prep' (ColumnTransformer)"
    assert "clf" in model.named_steps, "Le pipeline doit comporter l'étape 'clf' (Classifieur)"

    clf = model.named_steps["clf"]
    assert hasattr(clf, "decision_function") or hasattr(clf, "predict_proba"), (
        "Le classifieur doit fournir une marge ou probabilité de décision"
    )


def test_real_preprocessor_artifact(models_dir: Path):
    """Vérifie que preprocessor.pkl est désérialisable et possède transform."""
    prep_path = models_dir / "preprocessor.pkl"
    prep = joblib.load(prep_path)
    assert hasattr(prep, "transform"), "Le préprocesseur doit posséder la méthode transform"


@pytest.mark.parametrize("csv_name, expected_columns", [
    ("results.csv", ["Modèle", "Accuracy", "F1 macro", "ROC-AUC"]),
    ("profile_only_results.csv", ["Modèle", "Accuracy", "F1 macro"]),
    ("ablation.csv", ["Scénario", "Accuracy", "F1 macro"]),
    ("diagnostic.csv", ["n_key", "Employed=0", "Employed=1"]),
    ("fairness.csv", ["attribut", "taux réel", "accuracy"]),
])
def test_load_result_csv_real_tables(models_dir: Path, csv_name: str, expected_columns: list[str]):
    """Vérifie la lecture des tableaux de résultats et la conformité de leurs colonnes."""
    df = load_result_csv(models_dir, csv_name)
    assert df is not None, f"load_result_csv a retourné None pour {csv_name}"
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0, f"Le fichier {csv_name} est vide"
    for col in expected_columns:
        assert col in df.columns, f"Colonne '{col}' absente du fichier {csv_name}"


def test_load_result_csv_handles_missing_file(tmp_path: Path):
    """Vérifie qu'un fichier absent retourne None sans lever d'exception."""
    result = load_result_csv(tmp_path, "inexistent.csv")
    assert result is None


def test_load_result_csv_rejects_corrupted_file(tmp_path: Path):
    """Vérifie qu'un fichier CSV malformé lève ArtifactLoadError."""
    bad_csv = tmp_path / "corrupted.csv"
    # Écriture d'octets invalides (non décodables UTF-8)
    bad_csv.write_bytes(b"\x80\x81\xff\xfe\x00")
    with pytest.raises(ArtifactLoadError, match="illisible"):
        load_result_csv(tmp_path, "corrupted.csv")


def test_load_model_and_meta_rejects_corrupted_model_file(tmp_path: Path):
    """Vérifie qu'un fichier best_model.pkl corrompu lève ArtifactLoadError."""
    (tmp_path / "best_model.pkl").write_bytes(b"not a valid joblib/pickle file")
    (tmp_path / "meta.json").write_text(
        json.dumps({"champion": "Dummy", "technologies": ["Python"], "countries": ["France"]}),
        encoding="utf-8",
    )
    with pytest.raises(ArtifactLoadError, match="illisible ou incompatible"):
        load_model_and_meta(tmp_path)


def test_load_model_and_meta_rejects_model_without_predict(tmp_path: Path):
    """Vérifie qu'un objet sérialisé sans méthode predict lève ArtifactLoadError."""
    # Sauvegarde d'un dictionnaire à la place d'un modèle scikit-learn
    joblib.dump({"key": "value"}, tmp_path / "best_model.pkl")
    (tmp_path / "meta.json").write_text(
        json.dumps({"champion": "Dummy", "technologies": ["Python"], "countries": ["France"]}),
        encoding="utf-8",
    )
    with pytest.raises(ArtifactLoadError, match="ne fournit pas la méthode predict"):
        load_model_and_meta(tmp_path)

