"""Candidate-input validation and transparent prediction presentation helpers."""
from __future__ import annotations

from typing import Any

KEY_TECHS = frozenset({"Microsoft SQL Server", "npm", "jQuery", "TypeScript", "MongoDB", "SQLite", "Bash/Shell", "Node.js", "C#", "Java"})


def validate_candidate(years_code: int, years_pro: int, salary: float, technologies: list[str]) -> list[str]:
    errors: list[str] = []
    if years_pro > years_code:
        errors.append("Les années d'expérience professionnelle ne peuvent pas dépasser les années de pratique du code.")
    if salary < 0:
        errors.append("Le salaire précédent ne peut pas être négatif.")
    if len(technologies) != len(set(technologies)):
        errors.append("Chaque technologie ne doit être sélectionnée qu'une seule fois.")
    return errors


def key_technologies(technologies: list[str]) -> list[str]:
    return sorted(set(technologies).intersection(KEY_TECHS))


def deterministic_rule_summary(technologies: list[str]) -> dict[str, Any]:
    keys = key_technologies(technologies)
    return {"key_techs": keys, "count": len(keys), "rule_met": len(keys) >= 3}


def decision_margin_text(score: float) -> str:
    direction = "classe 1" if score >= 0 else "classe 0"
    return f"Marge de décision Linear SVC : {score:+.2f} (côté {direction}). Ce n'est pas une probabilité ni un niveau de confiance calibré."
