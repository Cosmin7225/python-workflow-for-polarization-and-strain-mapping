# Python Workflow for Polarization and Strain Mapping

A reproducible Python workflow for automated extraction of **polarization vector maps** and **directional lattice strain maps** from atomic-resolution STEM images.

The pipeline uses open-source scientific Python tools (HyperSpy, Atomap) to detect atomic columns, refine positions via 2D Gaussian fitting, reconstruct polarization vector fields from relative sublattice displacements, and compute real-space engineering strain from local interplanar spacings.

Designed for atomic-resolution studies of ferroelectric and functional oxide materials; adaptable to different crystal structures by adjusting detection parameters and the reference lattice spacing d₀.

---

## Repository Structure

```
Polarization_vector_map.ipynb   # polarization vector mapping workflow
Strain_map.ipynb                # directional lattice strain mapping workflow
environment.yml                 # conda environment with all dependencies
```

---

## Method Overview

### Polarization mapping (`Polarization_vector_map.ipynb`)

1. Image loading and spatial calibration (HyperSpy)
2. A-site atomic column detection and sub-pixel refinement (2D Gaussian fitting, Atomap)
3. Lattice vector estimation from nearest-neighbour statistics
4. B-site candidate generation and refinement (after A-site subtraction)
5. Zone-axis construction from the A sublattice
6. Polarization vectors from relative A–B sublattice displacements
7. Statistical descriptors: magnitude and angular distributions (MAD-based outlier rejection)
8. Visualization: quiver overlays, contour/colorwheel maps (TEMUL + Matplotlib)

### Strain mapping (`Strain_map.ipynb`)

1. Image loading and calibration
2. Atomic column detection and refinement
3. Zone-axis construction to define crystallographic directions
4. Real-space strain ε = (d − d₀) / d₀ from nearest-neighbour projections onto selected zone-axis directions
5. User-defined reference spacing d₀ (no unstrained reference region required)
6. Visualization: atom-resolved scatter maps and interpolated triangulation heat maps

---

## Requirements

Tested with Python 3.9–3.11.

| Package | Purpose |
|---------|---------|
| numpy, scipy | numerical operations |
| matplotlib | visualization |
| hyperspy | STEM image loading and calibration |
| atomap | atomic column detection and refinement |
| temul | polarization vector visualization |
| seaborn | statistical plots |
| colorcet | perceptually uniform colormaps (optional) |

---

## Installation

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate stem-pol
```

### Manual

```bash
conda create -n stem-pol python=3.10
conda activate stem-pol
pip install numpy scipy matplotlib seaborn jupyter
pip install hyperspy atomap temul
pip install colorcet
```

---

## Quick Start

```bash
git clone https://github.com/cosmin7225/python-workflow-for-polarization-and-strain-mapping.git
cd python-workflow-for-polarization-and-strain-mapping
conda env create -f environment.yml
conda activate stem-pol
jupyter lab
```

In each notebook, set `IMAGE_PATH` and `sampling` (nm/px) at the top of the configuration cell, then run all cells sequentially.

---

## Exporting Figures

```python
fig = plt.gcf()
fig.savefig("figure.png", dpi=400, bbox_inches="tight", facecolor="white")
fig.savefig("figure.pdf", bbox_inches="tight")
```

---

## Associated Publication

If you use this workflow in your research, please cite:

> M. C. Istrate, R. F. Negrea, "Automated polarization vector and strain mapping from atomic-resolution STEM images using a reproducible Python workflow," *Ultramicroscopy* (under review, 2025).

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

- Open a GitHub issue for questions or bug reports
- Email: cosmin.istrate@infim.ro
