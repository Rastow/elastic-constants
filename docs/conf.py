"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the
documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

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

html_theme = "pydata_sphinx_theme"
html_theme_options = {"use_edit_page_button": True}
# html_static_path = ["_static"]
html_context = {
    "github_user": "Rastow",
    "github_repo": "elastic-constants",
    "github_version": "master",
    "doc_path": "docs",
}
# https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/version-dropdown.html
# https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/readthedocs.html
# https://github.com/pydata/pydata-sphinx-theme/blob/460545510f581b3bf9ce34ddee5501949ba1b2b7/docs/conf.py#L131

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
