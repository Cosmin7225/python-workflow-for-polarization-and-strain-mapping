# polarmap

**Atomic-column displacement and real-space strain mapping from atomic-resolution
STEM images of ferroelectric perovskites.**

[![tests](https://github.com/Cosmin7225/python-workflow-for-polarization-and-strain-mapping/actions/workflows/tests.yml/badge.svg)](https://github.com/Cosmin7225/python-workflow-for-polarization-and-strain-mapping/actions/workflows/tests.yml)
[![docs](https://github.com/Cosmin7225/python-workflow-for-polarization-and-strain-mapping/actions/workflows/docs.yml/badge.svg)](https://cosmin7225.github.io/python-workflow-for-polarization-and-strain-mapping/)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![python](https://img.shields.io/badge/python-3.10%E2%80%933.14-blue)

`polarmap` is an installable, tested Python package that turns a calibrated
atomic-resolution image (HAADF, ADF, iDPC or ABF) into

- **displacement maps** of polar columns relative to their centrosymmetric
  reference, built from the *measured* neighbouring columns: complete
  four-corner cages (perovskite [100]), complete two-column pairs (perovskite
  [110]), or a fixed basis offset (dumbbell / wurtzite-type projections);
- **strain maps**: along one lattice direction against an external bulk spacing
  (Method 1), or the full in-plane strain and rigid-rotation tensor against a
  reference region of the same image (Method 2), both from a lattice graph whose
  bonds are validated by a loop-closure test;
- **statistics and publication figures**: circular orientation statistics,
  arrow maps and colour maps with scale bars, layer profiles, diagnostics.

> **Displacement is not polarization.** The measured quantity is an
> image-derived, projected displacement (pixels or pm). An absolute polarization
> would need the displacements of all sublattices, their Born effective-charge
> tensors and the unit-cell volume; `displacement_to_polarization` gives a
> clearly labelled single-sublattice estimate.

📖 **Documentation:** <https://cosmin7225.github.io/python-workflow-for-polarization-and-strain-mapping/>

## Installation

Python 3.10 or newer. The core needs only NumPy, SciPy and Matplotlib; the
`all` extra adds HyperSpy (for DM3/DM4/EMD files), colorcet and JupyterLab.

```bash
pip install "polarmap[all] @ git+https://github.com/Cosmin7225/python-workflow-for-polarization-and-strain-mapping.git"
```

or, with conda, from a clone of this repository:

```bash
conda env create -f environment.yml
conda activate polarmap
```

For development: `pip install -e ".[all,test]"` in a clone. See the
[installation guide](https://cosmin7225.github.io/python-workflow-for-polarization-and-strain-mapping/installation.html)
for all options.

## Quick example

```python
import polarmap as pm
from polarmap import plotting

image, sampling = pm.load_image("examples/data/PTO10040nmHAADF.dm4")   # nm/px from the file

seeds = pm.detect_peaks(image, min_distance=30, threshold_rel=0.2, exclude_border=5)
A = pm.refine_gaussian(image, seeds, box=10)          # sub-pixel Pb columns
v1, v2 = pm.estimate_lattice_vectors(A)
field = pm.measure_cage_displacement(image, A, v1, v2)  # Ti shift in each complete cage

print(pm.format_descriptors(pm.describe_displacements(field, sampling)))
# Displacement descriptors (n = 64, 0 excluded as outliers)
#   magnitude |d| (pm): mean 20.5  median 20.5 ...
plotting.plot_displacement_vectors(field, image, sampling=sampling, scalebar_nm="auto")
```

Strain (Methods 1 and 2) follows the same pattern:

```python
graph = pm.build_lattice_graph(positions, v1, v2)                       # validated bonds
m1 = pm.projection_strain(graph, v2, d0_px=0.3905 / sampling)           # vs bulk d0
reference = pm.fit_reference_lattice(graph, pm.in_rectangle(positions, (100, 1000), (450, 650)))
tensor = pm.tensor_strain(graph, reference)                             # exx, eyy, exy, omega
plotting.plot_strain_tensor(tensor, sampling=sampling)
```

## Examples

The notebooks in [`examples/`](examples) run the complete workflows on the data
in [`examples/data/`](examples/data) (rendered with outputs in the
documentation):

| Notebook | Content |
|---|---|
| [01_quickstart_synthetic](examples/01_quickstart_synthetic.ipynb) | recover a known displacement from a synthetic image (no files needed) |
| [02_displacement_perovskite_100](examples/02_displacement_perovskite_100.ipynb) | PbTiO₃ [100]: complete cages, statistics, arrow and colour maps |
| [03_displacement_perovskite_110](examples/03_displacement_perovskite_110.ipynb) | PbZrO₃ [110]: intensity split, complete pairs, antipolar pattern |
| [04_strain_superlattice](examples/04_strain_superlattice.ipynb) | experimental superlattice: lattice graph, Methods 1 and 2, interfaces |

## How it works

| Step | Module | Key idea |
|---|---|---|
| Load | `polarmap.io` | calibrated DM3/DM4/EMD (HyperSpy), MRC (built in), TIFF/PNG/NPY with explicit pixel size |
| Columns | `polarmap.columns` | local maxima + bounded 2-D Gaussian fit; Gaussian vs centre-of-mass precision proxy |
| Lattice | `polarmap.lattice` | lattice vectors from the columns; neighbours by minimum perpendicular offset; loop-closure validation; BFS indexing |
| Displacement | `polarmap.displacement` | reference = centroid of the measured complete cage, or interpolation along a measured complete pair |
| Strain | `polarmap.strain` | Method 1: $(d - d_0)/d_0$; Method 2: per-column deformation gradient $F$, $\varepsilon = \mathrm{sym}(F) - I$, $\omega = \mathrm{asym}(F)$ |
| Figures | `polarmap.plotting` | arrows or colour maps, colour wheels in the data's angle convention, scale bars, compact colour bars |

The [algorithm page](https://cosmin7225.github.io/python-workflow-for-polarization-and-strain-mapping/algorithm.html)
shows both workflows as flowcharts, and the
[limitations page](https://cosmin7225.github.io/python-workflow-for-polarization-and-strain-mapping/limitations.html)
discusses imaging artefacts, reference choices and interfaces.

### Coordinate conventions

Positions are `(x, y)` pixels with `+y` **down**. Displacements are
`(u, v) = measured target − reference`. Orientations use the Cartesian convention
`θ = atan2(−v, u)` (0° right, +90° up) for statistics *and* colour coding;
`convention="image"` switches to `atan2(v, u)`.

## Testing

```bash
pip install -e ".[test,io]"
pytest
```

The suite compares against exact ground truth from synthetic images and
lattices (`polarmap.synthetic`) and pins the results on the example data (for
instance 64 complete cages and a median Ti displacement of 20.545 pm for the
simulated PbTiO₃ image). Continuous integration runs it on Linux, macOS and
Windows for Python 3.10–3.14, with the oldest and newest supported NumPy, SciPy
and Matplotlib, and weekly against new releases.

## Repository layout

```text
src/polarmap/     the package
tests/            test suite
examples/         example notebooks and data
docs/             documentation (Sphinx, published to GitHub Pages)
legacy/           the original notebooks of the first manuscript version
```

## Citation

If you use `polarmap`, please cite it (see [`CITATION.cff`](CITATION.cff)); the
reference to the accompanying article will be added after publication.

## License and contact

MIT License (see [`LICENSE`](LICENSE)). Questions and bug reports:
[GitHub issues](https://github.com/Cosmin7225/python-workflow-for-polarization-and-strain-mapping/issues)
or cosmin.istrate@infim.ro.
