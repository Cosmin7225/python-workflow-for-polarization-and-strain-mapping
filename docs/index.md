# polarmap

**Atomic-column displacement and real-space strain mapping from
atomic-resolution STEM images of ferroelectric perovskites.**

`polarmap` is a tested, installable Python package that turns a calibrated
atomic-resolution image (HAADF, ADF, iDPC or ABF) into

* **displacement maps** of polar columns relative to their centrosymmetric
  reference: complete four-corner cages for perovskite [100], complete
  two-column pairs for perovskite [110], or a fixed basis offset for
  dumbbell-type (e.g. wurtzite) projections;
* **strain maps**, either along one lattice direction against an external bulk
  spacing (Method 1), or as the full in-plane strain and rigid-rotation tensor
  against a reference region of the same image (Method 2);
* **statistics and publication figures**: magnitude and circular orientation
  statistics, arrow maps, colour maps, layer profiles, diagnostics.

```{important}
The quantity measured from the image is a **projected displacement** (pixels or
picometres). It is not an absolute polarization: that would require the
displacements of all sublattices (including oxygen), their Born effective-charge
tensors and the unit-cell volume. {func}`polarmap.displacement.displacement_to_polarization`
provides a clearly labelled, semi-quantitative single-sublattice estimate.
```

## At a glance

```python
import polarmap as pm

image, sampling = pm.load_image("PTO100.dm4")                  # nm per pixel from the file
A = pm.refine_gaussian(image, pm.detect_peaks(image, min_distance=30), box=10)
v1, v2 = pm.estimate_lattice_vectors(A)
field = pm.measure_cage_displacement(image, A, v1, v2)          # B shift in each A cage
print(pm.format_descriptors(pm.describe_displacements(field, sampling)))

from polarmap import plotting
fig, ax = plotting.plot_displacement_vectors(field, image, sampling=sampling,
                                             scalebar_nm="auto")
```

## Design principles

* **Independent and auditable.** The analysis needs only NumPy, SciPy and
  Matplotlib; every step is a documented function. HyperSpy is used only, and
  optionally, to read proprietary file formats.
* **Tested against ground truth.** Synthetic images with exactly known
  displacements and strains ({mod}`polarmap.synthetic`) are used by the test
  suite, which runs on Linux, macOS and Windows with the oldest and the newest
  supported NumPy, SciPy and Matplotlib.
* **Explicit about what is excluded.** Columns without a complete reference,
  rejected bonds, edge columns and outliers are counted and reported, never
  dropped silently.
* **Local rather than global.** Displacement references are built from the
  measured neighbouring columns, and strain from each column's own bonds, so
  errors cannot propagate across the image.

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
quickstart
examples
```

```{toctree}
:maxdepth: 2
:caption: User guide

guide/loading
guide/columns
guide/displacement
guide/strain
guide/visualization
guide/validation
guide/batch
```

```{toctree}
:maxdepth: 1
:caption: Background

algorithm
conventions
limitations
migration
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
development
changelog
```
