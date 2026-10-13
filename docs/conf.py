"""Sphinx configuration: the user guide (the README), the validation report and the API
reference. Build with ``python -m sphinx -W -b html docs docs/_build/html``."""

from acoustic_spaces import __version__

project = "acoustic_spaces"
copyright = "acoustic_spaces contributors, GPL-3.0-or-later"
release = __version__
version = ".".join(__version__.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]
source_suffix = {".md": "markdown"}
exclude_patterns = ["_build"]

# Markdown: anchors for headings, so links like (#measuring-impulse-responses) resolve.
myst_heading_anchors = 3
myst_enable_extensions = ["dollarmath", "tasklist"]

autodoc_member_order = "bysource"
autodoc_typehints = "signature"
autodoc_default_options = {
    "members": True,
    "exclude-members": "model_config",
}

html_theme = "furo"
html_title = f"acoustic_spaces {release}"
html_static_path: list[str] = []
