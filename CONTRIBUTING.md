# Contributing

Bug reports, questions and suggestions are welcome as
[GitHub issues](https://github.com/Cosmin7225/python-workflow-for-polarization-and-strain-mapping/issues).
Please include the `polarmap` version (`python -c "import polarmap; print(polarmap.__version__)"`),
your operating system and, if possible, a small image or a synthetic example
(`polarmap.synthetic`) that reproduces the problem.

## Development setup

```bash
git clone https://github.com/Cosmin7225/python-workflow-for-polarization-and-strain-mapping.git
cd python-workflow-for-polarization-and-strain-mapping
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,io]"
```

## Before opening a pull request

```bash
ruff check src tests        # style and common errors
pytest                      # the full test suite (about a minute)
sphinx-build -W -b html docs docs/_build/html   # build the documentation
```

Every new function needs a NumPy-style docstring and a test. Tests should
compare against a known ground truth whenever possible: `polarmap.synthetic`
generates images and lattices with exactly known displacements and strains.
Tests that need HyperSpy or the example files are skipped automatically when
those are not available.

## Releasing

1. Update `src/polarmap/_version.py`, `CITATION.cff` and `CHANGELOG.md`.
2. Create a GitHub release with a tag such as `v0.1.0`. The `release`
   workflow builds the package and publishes it to PyPI; Zenodo (if enabled for
   the repository) archives the release and assigns a DOI.
