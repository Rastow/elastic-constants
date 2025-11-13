"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the
documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import re
import tomllib

from pathlib import Path


pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"

with pyproject_path.open("rb") as file:
    pyproject = tomllib.load(file)

# project information
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "elastic-constants"
copyright = "2025, Robert Grzonka"  # noqa: A001
author = "Robert Grzonka"
release = pyproject["project"]["version"]

# general configuration
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

version_requirements = {
    match.group(1): match.group(2)
    for version_requirement in pyproject["dependency-groups"]["docs"]
    if (match := re.match(r"^([^>=]+)>=([0-9.]+)$", version_requirement))
}
needs_sphinx = version_requirements.pop("sphinx")
extensions = [
    "sphinx.ext.apidoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_nb",
    "numpydoc",
    "sphinx_copybutton",
]
nitpicky = True
exclude_patterns = ["_build"]
source_suffix = {".rst": "restructuredtext", ".ipynb": "myst-nb"}

# configure the version switcher
json_url = "https://elastic-constants.readthedocs.io/en/latest/_static/switcher.json"
version_match = os.environ.get("READTHEDOCS_VERSION")
if not version_match or version_match.isdigit() or version_match == "latest":
    if "dev" in release or "rc" in release:
        version_match = "dev"
        json_url = "_static/switcher.json"
    else:
        version_match = release
elif version_match == "stable":
    version_match = release

# options for HTML output
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navbar_start": ["navbar-logo", "version-switcher"],
    "show_nav_level": 2,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/rastow/elastic-constants",
            "icon": "fa-brands fa-github",
        }
    ],
    "use_edit_page_button": True,
    "switcher": {"json_url": json_url, "version_match": version_match},
}
html_title = project
html_static_path = ["_static"]
html_context = {
    "github_user": "Rastow",
    "github_repo": project,
    "github_version": "master",
    "doc_path": "docs",
}

# apidoc options
apidoc_modules = [{"path": "../src/elastic_constants", "destination": "reference/"}]
apidoc_separate_modules = True

# autodoc options
autodoc_class_signature = "separated"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "fully-qualified"
autodoc_typehints_description_target = "documented"
autodoc_inherit_docstrings = False

# autosummary options
autosummary_generate = True

# intersphinx options
intersphinx_mapping = {
    "python": ("https://docs.python.org/3.12", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "spglib": ("https://spglib.readthedocs.io/en/stable/", None),
}

# myst options
myst_enable_extensions = ["dollarmath"]
myst_update_mathjax = False
myst_dmath_double_inline = True

# myst-nb options
nb_execution_mode = "cache"
