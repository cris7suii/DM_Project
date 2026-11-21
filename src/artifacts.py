"""Safe loading helpers for the versioned model artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class ArtifactLoadError(RuntimeError):
    """Raised when an application artifact is unavailable or incompatible."""


REQUIRED_META_KEYS = {"champion", "technologies", "countries"}


def load_model_and_meta(models_dir: Path) -> tuple[Any, dict[str, Any]]:
    """Load the champion pipeline and validate the metadata needed by the UI."""
    model_path = models_dir / "best_model.pkl"
    meta_path = models_dir / "meta.json"
    missing = [str(path.name) for path in (model_path, meta_path) if not path.exists()]
    if missing:
        raise ArtifactLoadError(f"Artefact(s) absent(s) : {', '.join(missing)}.")

    try:
        model = joblib.load(model_path)
    except Exception as exc:  # joblib exposes several version-specific exceptions
        raise ArtifactLoadError("Le modèle sérialisé est illisible ou incompatible avec cet environnement.") from exc

    if not hasattr(model, "predict"):
        raise ArtifactLoadError("Le modèle chargé ne fournit pas la méthode predict attendue.")

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactLoadError("Le fichier meta.json est illisible ou contient un JSON invalide.") from exc

    if not isinstance(meta, dict):
        raise ArtifactLoadError("meta.json doit contenir un objet JSON.")
    missing_keys = REQUIRED_META_KEYS.difference(meta)
    if missing_keys:
        raise ArtifactLoadError("meta.json ne contient pas les champs requis : " + ", ".join(sorted(missing_keys)) + ".")
    if not isinstance(meta["technologies"], list) or not isinstance(meta["countries"], list):
        raise ArtifactLoadError("Les listes technologies et countries de meta.json sont invalides.")
    return model, meta


def load_result_csv(models_dir: Path, name: str) -> pd.DataFrame | None:
    """Return an optional result table, raising a useful error when it is corrupt."""
    path = models_dir / name
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ArtifactLoadError(f"Le fichier de résultats {name} est illisible.") from exc
