import json
import sys
from pathlib import Path

import joblib
import pytest
from sklearn.dummy import DummyClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from artifacts import ArtifactLoadError, load_model_and_meta


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
