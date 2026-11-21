"""Configuration globale et fixtures pytest partagées pour DM_Project."""
import json
import sys
from pathlib import Path
import joblib
import pandas as pd
import pytest

# Garantit que 'src' est importable par tous les modules de test
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

KEY_TECHS = [
    "Microsoft SQL Server", "npm", "jQuery", "TypeScript", "MongoDB",
    "SQLite", "Bash/Shell", "Node.js", "C#", "Java"
]


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Retourne le chemin absolu vers la racine du projet."""
    return ROOT


@pytest.fixture(scope="session")
def models_dir(project_root: Path) -> Path:
    """Retourne le chemin absolu vers le dossier des artefacts de modèles."""
    return project_root / "models"


@pytest.fixture(scope="session")
def key_technologies() -> list[str]:
    """Retourne la liste des 10 technologies clés de la règle déterministe."""
    return list(KEY_TECHS)


@pytest.fixture(scope="session")
def loaded_artifacts(models_dir: Path):
    """Charge le modèle champion et les métadonnées meta.json."""
    model_path = models_dir / "best_model.pkl"
    meta_path = models_dir / "meta.json"
    if not model_path.exists() or not meta_path.exists():
        pytest.skip("Artefacts du modèle manquants dans models/")
    model = joblib.load(model_path)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


@pytest.fixture
def sample_candidate_dict() -> dict:
    """Dictionnaire représentatif d'un profil de candidat standard."""
    return {
        "Age": "<35",
        "Accessibility": "No",
        "EdLevel": "Master",
        "Employment": 1,
        "Gender": "Man",
        "MentalHealth": "No",
        "MainBranch": "Dev",
        "YearsCode": 8,
        "YearsCodePro": 5,
        "Country": "France",
        "PreviousSalary": 65000.0,
        "HaveWorkedWith": "Python;Node.js;TypeScript;Bash/Shell;Git",
        "ComputerSkills": 5
    }


@pytest.fixture
def sample_candidate_df(sample_candidate_dict: dict) -> pd.DataFrame:
    """DataFrame pandas contenant une ligne pour le candidat standard."""
    return pd.DataFrame([sample_candidate_dict])


@pytest.fixture
def sample_batch_df() -> pd.DataFrame:
    """DataFrame de plusieurs candidats couvrant différents profils et cas de la règle déterministe."""
    candidates = [
        # Candidat 1 : 4 technologies clés (>= 3 -> Employed = 1)
        {
            "Age": "<35", "Accessibility": "No", "EdLevel": "Master", "Employment": 1,
            "Gender": "Man", "MentalHealth": "No", "MainBranch": "Dev",
            "YearsCode": 10, "YearsCodePro": 6, "Country": "France", "PreviousSalary": 75000.0,
            "HaveWorkedWith": "Python;Node.js;TypeScript;Bash/Shell;MongoDB;Git", "ComputerSkills": 6
        },
        # Candidat 2 : 1 technologie clé (< 3 -> Not Employed = 0)
        {
            "Age": ">35", "Accessibility": "No", "EdLevel": "Undergraduate", "Employment": 1,
            "Gender": "Woman", "MentalHealth": "No", "MainBranch": "Dev",
            "YearsCode": 15, "YearsCodePro": 10, "Country": "Germany", "PreviousSalary": 80000.0,
            "HaveWorkedWith": "Python;Git;Docker;Node.js", "ComputerSkills": 4
        },
        # Candidat 3 : 0 technologie clé (< 3 -> Not Employed = 0)
        {
            "Age": "<35", "Accessibility": "Yes", "EdLevel": "PhD", "Employment": 0,
            "Gender": "NonBinary", "MentalHealth": "Yes", "MainBranch": "NotDev",
            "YearsCode": 3, "YearsCodePro": 1, "Country": "United States of America", "PreviousSalary": 45000.0,
            "HaveWorkedWith": "Python;R;Git", "ComputerSkills": 3
        },
        # Candidat 4 : 5 technologies clés (>= 3 -> Employed = 1)
        {
            "Age": ">35", "Accessibility": "No", "EdLevel": "Master", "Employment": 1,
            "Gender": "Man", "MentalHealth": "No", "MainBranch": "Dev",
            "YearsCode": 20, "YearsCodePro": 15, "Country": "United Kingdom", "PreviousSalary": 110000.0,
            "HaveWorkedWith": "C#;Java;Microsoft SQL Server;TypeScript;Node.js;AWS", "ComputerSkills": 6
        }
    ]
    return pd.DataFrame(candidates)
