"""Tests unitaires pour la logique de l'application Streamlit et les composants d'interface."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import simulation
import theme


# ==============================================================================
# Tests du générateur HTML du simulateur (src/simulation.py)
# ==============================================================================

def test_simulation_build_html_generates_valid_inlined_markup():
    """Vérifie que build_html assemble correctement le HTML, le CSS et le JS inlinés."""
    html_output = simulation.build_html()

    assert isinstance(html_output, str)
    assert len(html_output) > 2000, "Le HTML généré est anormalement court"

    # Vérifie la présence des balises structurales obligatoires
    assert "<html" in html_output
    assert "</html>" in html_output
    assert '<canvas id="cv"' in html_output
    assert '<nav id="nav"' in html_output
    assert '<div class="card" id="panel">' in html_output

    # Vérifie que les fichiers CSS et JS ont bien été inlinés dans <style> et <script>
    assert "<style>" in html_output
    assert "</style>" in html_output
    assert "<script>" in html_output
    assert "</script>" in html_output

    # Vérifie qu'aucun lien externe résiduel vers les fichiers locaux n'est resté
    assert '<link rel="stylesheet" href="css/style.css">' not in html_output
    assert '<script src="js/core.js"></script>' not in html_output


# ==============================================================================
# Tests des composants d'interface et du Design System (src/theme.py)
# ==============================================================================

def test_badge_formatting():
    """Vérifie la génération des balises HTML badge."""
    assert theme.badge("Python", "champion") == '<span class="badge champion">Python</span>'
    assert theme.badge("Test", "warn") == '<span class="badge warn">Test</span>'
    assert theme.badge("Normal") == '<span class="badge ">Normal</span>'


def test_inject_theme_renders_css(monkeypatch):
    """Vérifie que inject_theme transmet le CSS global à st.markdown."""
    mock_markdown = MagicMock()
    monkeypatch.setattr(st, "markdown", mock_markdown)

    theme.inject_theme()

    mock_markdown.assert_called_once()
    rendered_css = mock_markdown.call_args[0][0]
    assert "<style>" in rendered_css
    assert "--bg:" in rendered_css
    assert "--surface:" in rendered_css
    assert mock_markdown.call_args[1].get("unsafe_allow_html") is True


def test_page_header_rendering(monkeypatch):
    """Vérifie le rendu d'un en-tête de page avec et sans description."""
    rendered_blocks = []
    monkeypatch.setattr(st, "markdown", lambda content, **kw: rendered_blocks.append(content))

    # Avec description
    theme.page_header("ANALYSE", "Titre de Section", "Description détaillée.")
    assert len(rendered_blocks) == 1
    assert "ANALYSE" in rendered_blocks[0]
    assert "Titre de Section" in rendered_blocks[0]
    assert "Description détaillée." in rendered_blocks[0]
    assert 'class="page-desc"' in rendered_blocks[0]

    # Sans description
    rendered_blocks.clear()
    theme.page_header("MODÈLE", "Titre Court")
    assert len(rendered_blocks) == 1
    assert "MODÈLE" in rendered_blocks[0]
    assert 'class="page-desc"' not in rendered_blocks[0]


def test_hero_banner_rendering(monkeypatch):
    """Vérifie le composant hero d'accueil."""
    rendered_blocks = []
    monkeypatch.setattr(st, "markdown", lambda content, **kw: rendered_blocks.append(content))

    theme.hero("EmployML", "Prédire", "l'employabilité", "Sous-titre descriptif.")

    assert len(rendered_blocks) == 1
    html = rendered_blocks[0]
    assert "EmployML" in html
    assert "Prédire" in html
    assert "<em>l'employabilité</em>" in html
    assert "Sous-titre descriptif." in html
    assert 'class="hero-title"' in html


def test_warning_and_ethical_cards(monkeypatch):
    """Vérifie les cartes d'avertissement méthodologique et éthique."""
    rendered_blocks = []
    monkeypatch.setattr(st, "markdown", lambda content, **kw: rendered_blocks.append(content))

    theme.warning_card("Alerte Règle", "Détail de la règle déterministe.")
    theme.ethical_card("Avertissement Biais", "Détail de l'analyse d'équité.")

    assert len(rendered_blocks) == 2
    # Warning card
    assert "warning-card" in rendered_blocks[0]
    assert "Alerte Règle" in rendered_blocks[0]
    assert "Détail de la règle déterministe." in rendered_blocks[0]

    # Ethical card
    assert "ethical-card" in rendered_blocks[1]
    assert "Avertissement Biais" in rendered_blocks[1]
    assert "Détail de l'analyse d'équité." in rendered_blocks[1]


def test_result_card_positive_class(monkeypatch):
    """Vérifie l'affichage de la carte de résultat pour la classe Employé (1)."""
    rendered_blocks = []
    monkeypatch.setattr(st, "markdown", lambda content, **kw: rendered_blocks.append(content))

    theme.result_card(
        pred=1,
        score_text="Distance: +1.50",
        proba=0.82,
        key_techs_found=["Node.js", "TypeScript", "Bash/Shell"]
    )

    assert len(rendered_blocks) == 1
    html = rendered_blocks[0]
    assert "Candidat Employé (Classe 1)" in html
    assert "positive" in html
    assert "82%" in html
    assert "Distance: +1.50" in html
    assert "3 / 10 clés" in html
    assert "Règle déterministe déclenchée" in html


def test_result_card_negative_class(monkeypatch):
    """Vérifie l'affichage de la carte de résultat pour la classe Non Employé (0)."""
    rendered_blocks = []
    monkeypatch.setattr(st, "markdown", lambda content, **kw: rendered_blocks.append(content))

    theme.result_card(
        pred=0,
        score_text="Distance: -0.75",
        proba=0.32,
        key_techs_found=["Python"]
    )

    assert len(rendered_blocks) == 1
    html = rendered_blocks[0]
    assert "Candidat Non Employé (Classe 0)" in html
    assert "negative" in html
    assert "32%" in html
    assert "Distance: -0.75" in html
    assert "Seuil déterministe non atteint" in html


def test_pipeline_stepper_rendering(monkeypatch):
    """Vérifie le stepper de présentation des étapes du projet."""
    rendered_blocks = []
    monkeypatch.setattr(st, "markdown", lambda content, **kw: rendered_blocks.append(content))

    steps = [
        ("Nettoyage", "Imputations et outliers"),
        ("Prétraitement", "ColumnTransformer"),
        ("Modélisation", "GridSearchCV"),
    ]
    theme.pipeline(steps)

    assert len(rendered_blocks) == 1
    html = rendered_blocks[0]
    assert 'class="pipeline"' in html
    assert "<b>1</b><strong>Nettoyage</strong>" in html
    assert "<b>2</b><strong>Prétraitement</strong>" in html
    assert "<b>3</b><strong>Modélisation</strong>" in html


def test_metric_cards_rendering(monkeypatch):
    """Vérifie la mise en page des cartes KPI avec st.columns."""
    rendered_blocks = []
    monkeypatch.setattr(st, "markdown", lambda content, **kw: rendered_blocks.append(content))

    # Mock st.columns pour simuler un contexte Streamlit
    class MockColumn:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    monkeypatch.setattr(st, "columns", lambda n: [MockColumn() for _ in range(n)])

    items = [
        {"icon": "🏆", "value": "Linear SVC", "label": "Champion", "tone": "violet"},
        {"icon": "👥", "value": "53.6%", "label": "Taux", "tone": "green"},
    ]
    theme.metric_cards(items)

    assert len(rendered_blocks) == 2
    assert "Linear SVC" in rendered_blocks[0]
    assert "Champion" in rendered_blocks[0]
    assert "53.6%" in rendered_blocks[1]
    assert "Taux" in rendered_blocks[1]
