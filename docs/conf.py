"""Sphinx configuration for moire-metrology documentation."""

project = "moire-metrology"
author = "Dorri Halbertal"
copyright = "2026, Dorri Halbertal"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

# -- General -----------------------------------------------------------
import warnings
warnings.filterwarnings("ignore", message=".*RemovedInSphinx10Warning.*")

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Suppress warnings from scipy/numpy docstrings pulled in by autodoc and
# from cross-reference ambiguity on re-exported names.
suppress_warnings = [
    "ref.python",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- MyST (Markdown) ---------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "fieldlist",
]
myst_heading_anchors = 3

# -- Autodoc / Napoleon ------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "imported-members": False,
}
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Intersphinx -------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- HTML output -------------------------------------------------------
html_theme = "furo"
html_title = "moire-metrology"
html_theme_options = {
    "source_repository": "https://github.com/dorrih/moire_metrology",
    "source_branch": "main",
    "source_directory": "docs/",
}
