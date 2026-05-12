from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "tokamaker-jax"
author = "Rogerio Jorge"
copyright = "2026, Rogerio Jorge"
release = "0.1.0a0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
master_doc = "index"
html_theme = "furo"
html_static_path = ["_static"]
autodoc_typehints = "description"
myst_enable_extensions = ["colon_fence", "dollarmath"]
