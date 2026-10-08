# Development

## Layout of the repository

```text
src/polarmap/          the package
    io.py              image loading and calibration
    columns.py         detection and sub-pixel refinement
    lattice.py         lattice vectors, lattice graph, loop closure, BFS
    displacement.py    reference models and displacement fields
    strain.py          Method 1 and Method 2 strain
    statistics.py      conventions, circular and robust statistics
    validation.py      VecMap reader, comparison metrics
    synthetic.py       synthetic images and lattices with ground truth
    geometry.py        small geometric helpers
    plotting/          figures
tests/                 pytest suite (synthetic ground truth + example data)
examples/              example notebooks and data
docs/                  this documentation (Sphinx)
legacy/                the original notebooks, for reference
```

## Setting up

```bash
git clone https://github.com/Cosmin7225/python-workflow-for-polarization-and-strain-mapping.git
cd python-workflow-for-polarization-and-strain-mapping
pip install -e ".[dev,io]"
```

## Tests

```bash
pytest                              # all tests
pytest -m "not hyperspy"            # without the tests that need HyperSpy
pytest --cov=polarmap               # with coverage
```

Most tests compare against exact ground truth from {mod}`polarmap.synthetic`.
Regression tests on the example data pin published numbers (for instance the
64 complete cages and 20.545 pm median displacement of the simulated PbTiO₃
image), so that any change of behaviour is noticed. Warnings about deprecated
NumPy/SciPy/Matplotlib features raised from within `polarmap` are turned into
errors, so API changes of the dependencies surface as test failures.

Continuous integration (GitHub Actions, `.github/workflows/tests.yml`) runs the
suite on Linux, macOS and Windows for Python 3.10–3.14, once with the *oldest*
supported NumPy/SciPy/Matplotlib, once with HyperSpy and the example files, and
weekly against the newest releases.

## Documentation

```bash
sphinx-build -W --keep-going -b html docs docs/_build/html
```

The documentation is published to GitHub Pages by
`.github/workflows/docs.yml` on every push to `main` (enable it once under
*Settings → Pages → Source: GitHub Actions*). The example notebooks are
rendered with their stored outputs; re-run them after changing the code.

## Style

`ruff check src tests` (configuration in `pyproject.toml`). Docstrings follow the
NumPy convention.

## Releasing and distribution

1. Update the version in `src/polarmap/_version.py` and `CITATION.cff`, and the
   `CHANGELOG.md`.
2. Tag and publish a GitHub release (e.g. `v0.1.0`).
3. `.github/workflows/release.yml` builds the wheel and source distribution
   and uploads them to PyPI through *trusted publishing* (configure the pending
   publisher on PyPI once, as described in the workflow file).
4. With the Zenodo–GitHub integration enabled, each release is archived with a
   DOI, which can be cited in the paper.
5. A conda-forge package can then be proposed from the PyPI release
   (`grayskull pypi polarmap` generates the recipe for
   `conda-forge/staged-recipes`).
