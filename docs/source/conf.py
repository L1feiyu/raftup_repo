# Configuration file for the Sphinx documentation builder.

project = 'RAFT-UP'
copyright = '2025, YW'
author = 'YW'
release = '0.1.0'

extensions = [
    "myst_parser",
    "nbsphinx",
]

templates_path = ['_templates']
exclude_patterns = []

nbsphinx_execute = "never"

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
