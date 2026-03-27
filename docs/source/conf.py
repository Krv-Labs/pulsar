import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "Pulsar"
author = "Krv Labs"
copyright = "2026, Krv Labs"
release = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.todo",
    "sphinx.ext.githubpages",
    "sphinxcontrib.mermaid",
]

autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2
source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
html_title = "Pulsar Documentation"
html_theme_options = {
    "light_logo": "logo.png",
    "dark_logo": "logo-dark.png",
}

html_css_files = ["custom.css"]
html_js_files = ["logo-link.js"]

nbsphinx_allow_errors = True
