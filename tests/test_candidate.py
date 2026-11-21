import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from candidate import decision_margin_text, deterministic_rule_summary, validate_candidate


def test_candidate_validation_rejects_inconsistent_experience():
    assert validate_candidate(2, 3, 1000, ["Python"])


def test_deterministic_rule_counts_only_documented_technologies():
    summary = deterministic_rule_summary(["Python", "Node.js", "TypeScript", "Bash/Shell"])
    assert summary["count"] == 3
    assert summary["rule_met"] is True


def test_margin_description_never_claims_probability():
    assert "n'est pas une probabilité" in decision_margin_text(0.4)
