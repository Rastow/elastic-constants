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
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "numpydoc",
    "sphinx_copybutton",
]
exclude_patterns = ["_build"]
nitpicky = True

# options for HTML output
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# configure version switcher, see:
# https://github.com/pydata/pydata-sphinx-theme/blob/460545510f581b3bf9ce34ddee5501949ba1b2b7/docs/conf.py#L130
json_url = "https://elastic-constants.readthedocs.io/en/latest/_static/switcher.json"
version_match = os.environ.get("READTHEDOCS_VERSION")
if not version_match or version_match.isdigit() or version_match == "latest":
    if "-" in release:
        version_match = "dev"
        json_url = "_static/switcher.json"
    else:
        version_match = release
elif version_match == "stable":
    version_match = release

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/rastow/elastic-constants",
            "icon": "fa-brands fa-github",
        },
    ],
    "use_edit_page_button": True,
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "switcher": {
        "json_url": json_url,
        "version_match": version_match,
    },
}
html_title = f"{project} v{release} Manual"
html_static_path = ["_static"]
html_context = {
    "github_user": "Rastow",
    "github_repo": "elastic-constants",
    "github_version": "master",
    "doc_path": "docs",
}

# apidoc options
apidoc_modules = [{"path": "../src/elastic_constants", "destination": "reference/"}]
apidoc_exclude_patterns = []
apidoc_follow_links = True
apidoc_separate_modules = True
apidoc_module_first = False
apidoc_implicit_namespaces = True

# autodoc options
autodoc_member_order = "bysource"
autodoc_typehints = "both"
autodoc_typehints_format = "fully-qualified"
autodoc_inherit_docstrings = False

# intersphinx options
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "spglib": ("https://spglib.readthedocs.io/en/stable/", None),
}

# nbsphinx
nbsphinx_allow_errors = True
nbsphinx_execute = "always"

# numpydoc
numpydoc_validation_checks = {"all"}
