# Detecting and refining atomic columns

Column positions are obtained in two steps: integer-pixel **seeds** from a
local-maximum detector, then **sub-pixel refinement** by fitting a 2-D Gaussian
to each column.

## 1. Seeds: local maxima

```python
seeds = pm.detect_peaks(image, min_distance=30, threshold_rel=0.2,
                        exclude_border=5, image_mode="HAADF")
```

{func}`polarmap.columns.detect_peaks` proceeds as follows:

1. **Background flattening** (`flatten=True`): a Gaussian-blurred copy of the
   image (standard deviation `bg_sigma`, default `2 * min_distance`) is
   subtracted and the result rescaled to [0, 1]. This removes slow intensity
   variations (illumination, thickness gradients, the low-frequency background
   of iDPC) so that one global threshold works across the image.
2. **Maximum filter**: a pixel is a peak if it equals the maximum of the
   `min_distance` x `min_distance` window centred on it.
3. **Threshold**: peaks below `threshold_rel` of the intensity range are
   discarded.
4. **Ties**: equal neighbouring maxima (frequent in 8-bit or saturated images)
   are merged into their centroid, so each column yields one seed.

### Choosing `min_distance`

`min_distance` selects *which sublattice* is detected:

* set it close to the **spacing of the sublattice you want**. For the A sites
  of a perovskite [100] image, use the A–A spacing: the weaker B columns, only
  `a/√2` away, fall inside the window of the brighter A columns and are
  suppressed;
* a smaller value detects all columns (e.g. both sublattices of a [110] image,
  which are then separated by intensity, see below).

If you do not know the spacing, measure it on the image (e.g. a line profile
or the FFT) or start with a small value, inspect the detected columns, and use
{func}`polarmap.lattice.estimate_lattice_vectors` to measure the spacing.

```{tip}
Always inspect the seeds with {func}`polarmap.plotting.diagnostics.plot_columns`: every
intended column should carry exactly one marker, and the other sublattice none.
If the detected set mixes sublattices, the lattice vectors come out wrong and
many bonds fail the loop-closure test.
```

## 2. Sub-pixel refinement: 2-D Gaussian fit

```python
A, fit = pm.refine_gaussian(image, seeds, box=10, return_params=True)
A = A[fit["success"]]
```

{func}`polarmap.columns.refine_gaussian` fits, in a square window of
`(2 box + 1)` pixels around each seed, the elliptical Gaussian

$$
I(x, y) = A\,\exp\!\left[-\tfrac{1}{2}\left(\frac{x_r^2}{\sigma_x^2}
          + \frac{y_r^2}{\sigma_y^2}\right)\right] + B ,
$$

where $(x_r, y_r)$ are the coordinates relative to the centre $(x_0, y_0)$
rotated by $\theta$. The seven parameters are found by bounded non-linear least
squares (trust-region reflective, `scipy.optimize.least_squares`, analytic
Jacobian). The bounds keep the centre inside the window, the widths between
0.5 px and `3 box`, and the amplitude positive.

* `box` should cover the column but not reach its neighbours; about one third
  of the nearest-neighbour spacing works well.
* Where the window leaves the image or the fit fails, the seed is returned
  unchanged and `fit["success"]` is `False`; discard those columns as shown.
* `fit` also contains `amplitude`, `sigma_x`, `sigma_y` and `ellipticity`
  (the ratio of the two widths) for quality control.

{func}`polarmap.columns.refine_com` is a faster centre-of-mass alternative, biased when
neighbouring columns enter the window.

## A precision proxy

The Gaussian fit and the centre of mass make different assumptions about the
column shape. Their disagreement is a practical measure of how well the
positions are defined:

```python
pm.compare_refinements(image, seeds, sampling, box=10)
# {'rms_pm': 2.2, 'median_pm': 2.1, ...}      (simulated PbTiO3 [100])
```

This is a *precision* estimate, not an accuracy estimate: effects that shift
the apparent column position for both estimators alike (mistilt, residual
aberrations, channelling, scan distortions) are not captured. For accuracy,
analyse synthetic images with known positions ({mod}`polarmap.synthetic`) or
multislice simulations of your structure at your imaging conditions.

## Two sublattices of similar intensity

When bright and weak columns must both be detected (e.g. Pb and Ti–O columns in
a perovskite [110] projection), detect all columns with a small
`min_distance`, then split them by fitted amplitude:

```python
seeds = pm.detect_peaks(image, min_distance=8, threshold_rel=0.2)
split = pm.split_by_amplitude(image, seeds, box=3, edge=10)
split.bright, split.dim           # refined positions of both groups
```

The threshold between the two groups is found by iterative 1-D two-means
clustering of the amplitudes ({func}`polarmap.columns.two_means_threshold`).

## Positions from another program

All downstream functions take plain `(N, 2)` arrays of `(x, y)` pixel
positions, so positions refined with any other tool (e.g. Atomap) can be used
directly:

```python
A = np.asarray(atomap_sublattice.atom_positions)     # (N, 2), x then y
```
