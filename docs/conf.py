"""Sphinx configuration for the openqcv reference documentation."""

import openqcv

project = "openqcv"
author = "Keno-00"
copyright = "2026, Keno-00"
release = openqcv.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
autodoc_member_order = "bysource"
autodoc_typehints = "description"

exclude_patterns = ["_build"]

html_theme = "groundwork"
