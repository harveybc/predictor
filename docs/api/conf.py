"""Sphinx configuration for the modular temporal API reference."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

project = "Predictor Modular Temporal API"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_inherit_docstrings = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
exclude_patterns = ["_build"]
root_doc = "index"
html_theme = "alabaster"
