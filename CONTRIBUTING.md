# Contributing

Read [docs/design.md](docs/design.md) before changing code.

## Setup

```sh
git clone https://github.com/Keno-00/open-quantum-computer-vision.git
cd open-quantum-computer-vision
pip install -e ".[test,lint,docs]"
```

## Checks

Run from the repository root. All four must pass before a change is done.

```sh
ruff format --check .
ruff check .
pytest
sphinx-build -W --keep-going -b html docs docs/_build/html
```

The last command builds the reference documentation from the docstrings
into `docs/_build/html`; open `index.html` there to read it.

If a change affects the figures in `docs/images`, regenerate them:

```sh
python docs/make_figures.py
```

## Pull requests

1. Fork the repository and create a branch (`git checkout -b feature/name`).
2. Make the change, with tests, and run the checks.
3. Commit (`git commit -m "Short imperative summary"`) and push the branch.
4. Open a pull request that says what changed and why.
