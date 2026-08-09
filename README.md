# Atomic-column displacement and real-space strain mapping from atomic-resolution STEM images

This repository contains my transparent Python workflow for extracting **projected atomic-column displacements** and real-space strain from atomic-resolution STEM images (HAADF/ABF).

The quantity I measure directly is an image-derived displacement in pixels, nanometres, or picometres. It is **not an absolute polarization** unless the displacements of all relevant sublattices, their Born effective-charge tensors, and the unit-cell volume are supplied.

## Scope

The workflow provides:

- calibrated loading of DM3/DM4, EMD, HDF5 and other formats supported by HyperSpy;
- independent local-maximum detection and bounded two-dimensional Gaussian refinement;
- complete four-corner cage references for perovskite-like projections such as [100];
- complete pair references for projections such as perovskite [110];
- a fractional pair reference that can be adapted to a side-view wurtzite geometry;
- displacement-vector, magnitude and orientation maps;
- circular statistics for vector orientations;
- directional strain relative to an external reference spacing;
- full in-plane strain and rigid-body rotation from a validated local lattice graph;
- diagnostic plots, outlier reporting and portable figure export.

The implementation uses NumPy, SciPy, Matplotlib, pandas and HyperSpy, with Atomap used for atomic-column detection in the strain routine. Trusted coordinates refined by another program can also be supplied directly to the reference and strain routines.

## Repository structure

```text
notebooks/
    Displacement_workflow_complete.ipynb   # atomic-column displacement mapping
    Strain_map_python_routine.ipynb        # real-space strain and rotation mapping

example images/
    ImgOrigin.dm4                          # raw acquired image
    PZO11040nmHAADF.dm4                     # PbZrO3-type HAADF example
    PTO10040nmHAADF.dm4                     # PbTiO3-type HAADF example

README.md
requirements.txt
```

- **`Displacement_workflow_complete.ipynb`** detects atomic columns, builds the reference sublattice, and computes the per-column displacement vectors, magnitude and orientation maps.
- **`Strain_map_python_routine.ipynb`** takes the same calibrated images and reconstructs the in-plane strain tensor and rigid-body rotation from a validated local lattice graph.

The files in **`example images/`** are the calibrated DM4 datasets I use to run the notebooks end to end.

## Crystallographic reference models

### Complete four-corner cage

For a perovskite-like [100] projection, the ideal target position is the centroid of the four **measured** reference columns forming a complete projected cell. I reject a target site if any corner is missing or outside the geometric tolerance.

### Complete pair

For a perovskite [110] projection, the ideal target position is the midpoint between two measured, crystallographically equivalent reference columns:

```text
r_ideal = 0.5 r_0 + 0.5 r_1
```

For a side-view wurtzite projection, the same pair implementation can be used with the ideal fractional coordinate `u_ref = 3/8` along a complete local cation-to-cation repeat:

```text
r_N,ideal = (1 - u_ref) r_C,k + u_ref r_C,k+1
```

This wurtzite adaptation requires structure- and imaging-specific validation. In particular, light-column positions may be affected by mistilt, residual aberrations, thickness-dependent channeling and overlap with neighbouring heavy columns.

## Coordinate convention

Image coordinates use `+x` to the right and `+y` downward. The stored displacement is

```text
(u, v) = measured_target - local_reference
```

For Cartesian orientation statistics and colour coding, I use

```text
theta = atan2(-v, u)
```

so that `0°` points rightward, `+90°` points upward and positive angles increase counter-clockwise.

## Installation

I recommend creating a clean environment and installing the pinned dependencies:

```bash
conda create -n stem-displacement python=3.10
conda activate stem-displacement
pip install -r requirements.txt
```

`colorcet` is optional; Matplotlib's cyclic colormap is used as a fallback.

## Running the notebooks

Launch Jupyter and open the notebooks in the **`notebooks/`** folder:

```bash
jupyter lab
```

For experimental data, set the image path to the original calibrated DM3/DM4, EMD or HDF5 file whenever possible. The datasets in **`example images/`** can be used directly to reproduce the workflow. Display-oriented PNG/JPEG/TIFF files should be used only as a fallback and require an explicitly verified pixel calibration.

Run each notebook from top to bottom. Before interpreting a map, inspect the diagnostic overlays and confirm that:

1. reference-column seeds identify only the intended sublattice;
2. every cage reference has four measured corners;
3. every pair reference has both measured endpoints;
4. target fits remain within the local search window;
5. vector sign and angle conventions match the manuscript;
6. edge and outlier exclusions are reported.

## Validation

The associated manuscript validates the displacement results against Vec-map for perovskite [100] and [110] projections and compares the strain maps with geometric phase analysis using the same calibrated image, reference region and matched effective spatial resolution. Numerical comparison tables and processing-sensitivity tests are provided in the Supplementary Information.

## Reproducibility

All analysis steps are implemented directly in the notebooks and are intended to be reproducible. Key processing parameters are explicitly defined in the workflow. A tagged release and an archived DOI are recommended for the manuscript version.

## Citation

Citation information will be added after publication (manuscript in preparation).

## License

MIT License.

## Contact

For questions or collaboration requests, please open a GitHub issue or contact me by e-mail (cosmin.istrate@infim.ro).
