# Installation

`polarmap` requires Python 3.10 or newer. Its only mandatory dependencies are
NumPy, SciPy and Matplotlib.

## With pip

Install the latest version directly from GitHub, including HyperSpy for
reading DM3/DM4/EMD files and colorcet for a perceptually uniform colour wheel:

```bash
pip install "polarmap[all] @ git+https://github.com/Cosmin7225/python-workflow-for-polarization-and-strain-mapping.git"
```

Once the package is published on PyPI this becomes simply
`pip install "polarmap[all]"`.

The optional extras can be combined as needed:

| Extra | Installs | Needed for |
|---|---|---|
| *(none)* | NumPy, SciPy, Matplotlib | all analysis and figures; MRC, TIFF, PNG, JPG and NPY input |
| `io` | HyperSpy | DM3/DM4, Velox EMD, HSPY/HDF5, SER and other microscopy formats |
| `colors` | colorcet | the `cet_colorwheel` cyclic colour map (otherwise `hsv`) |
| `all` | `io` + `colors` + JupyterLab | typical interactive use |
| `test` | pytest | running the test suite |
| `docs` | Sphinx and extensions | building this documentation |

## With conda

A conda environment with every dependency from conda-forge:

```bash
git clone https://github.com/Cosmin7225/python-workflow-for-polarization-and-strain-mapping.git
cd python-workflow-for-polarization-and-strain-mapping
conda env create -f environment.yml
conda activate polarmap
```

## From a local clone (for development)

```bash
git clone https://github.com/Cosmin7225/python-workflow-for-polarization-and-strain-mapping.git
cd python-workflow-for-polarization-and-strain-mapping
pip install -e ".[all,test]"
```

The `-e` (editable) flag makes changes to the source take effect without
reinstalling.

## Check the installation

```bash
python -c "import polarmap; print(polarmap.__version__)"
pytest                     # from the repository root (needs the "test" extra)
```

The first example notebook, `examples/01_quickstart_synthetic.ipynb`, needs no
data files and verifies that a known displacement is recovered.
