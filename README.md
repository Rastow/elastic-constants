# elastic-constants

## Installation
The package is available on the 
[Python Package Index](https://pypi.org/project/elastic-constants/)
and can be installed using the following command.
```bash
pip install elastic-constants
```

## Features
This library provides tools for computing the mechanical properties of
crystal structures. Currently, only the computation of linear elastic 
constants is provided for all crystal classes. Additionally, the library
includes a useful visualization module.

## Documentation
The [documentation](https://rastow.github.io/elastic-constants) is made
with [Sphinx](https://www.sphinx-doc.org/en/master/)
and is hosted on [GitHub Pages](https://docs.github.com/en/pages).

## Development
This project uses [uv](https://docs.astral.sh/uv/) as the package manager
and [tox](https://tox.wiki/en/stable/) as a task runner. After 
installing uv and cloning the repository, create a development 
environment using the following commands.
```bash
uv tool install tox --with tox-uv
uv sync --all-extras --all-groups
```

## License
The project is distributed under the terms of the 
[BSD-2-Clause](https://spdx.org/licenses/BSD-2-Clause.html) license.