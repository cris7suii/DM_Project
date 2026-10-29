"""Génère le simulateur HTML autonome (CSS + JS inlinés) pour l'onglet Streamlit « Simulation »."""
import re
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parents[1] / "simulation"


def build_html() -> str:
    html = (SIM_DIR / "index.html").read_text(encoding="utf-8")

    def inline_js(m):
        f = SIM_DIR / m.group(1)
        return f"<script>\n{f.read_text(encoding='utf-8')}\n</script>" if f.exists() else ""

    def inline_css(m):
        return f"<style>\n{(SIM_DIR / m.group(1)).read_text(encoding='utf-8')}\n</style>"

    html = re.sub(r'<script src="([^"]+)"></script>', inline_js, html)
    return re.sub(r'<link rel="stylesheet" href="([^"]+)">', inline_css, html)
