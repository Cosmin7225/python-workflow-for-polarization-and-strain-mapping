"""Sphinx configuration for the polarmap documentation.

Build locally with::

    pip install -e ".[docs]"
    sphinx-build -W --keep-going -b html docs docs/_build/html

and open ``docs/_build/html/index.html``.
"""

import shutil
from pathlib import Path

import polarmap

HERE = Path(__file__).resolve().parent

# -- Project information -----------------------------------------------------
project = "polarmap"
author = "Marian Cosmin Istrate and Raluca Florentina Negrea"
copyright = "2026, Marian Cosmin Istrate"
release = polarmap.__version__
version = release

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "myst_nb",
    "sphinxcontrib.mermaid",
    "sphinx_copybutton",
]

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_fence_as_directive = ["mermaid"]
myst_heading_anchors = 3
nb_execution_mode = "off"          # show the outputs stored in the notebooks

autodoc_member_order = "bysource"
autodoc_typehints = "none"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

exclude_patterns = ["_build", "**.ipynb_checkpoints"]
suppress_warnings = ["mystnb.unknown_mime_type"]

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_title = f"polarmap {release}"
html_theme_options = {
    "source_repository": "https://github.com/Cosmin7225/python-workflow-for-polarization-and-strain-mapping/",
    "source_branch": "main",
    "source_directory": "docs/",
}

# -- Example notebooks -------------------------------------------------------
# The notebooks live in ../examples so that they can be run from a checkout;
# copy them next to the documentation sources for rendering.
_examples_src = HERE.parent / "examples"
_examples_dst = HERE / "examples"
_examples_dst.mkdir(exist_ok=True)
for notebook in sorted(_examples_src.glob("*.ipynb")):
    shutil.copy2(notebook, _examples_dst / notebook.name)
