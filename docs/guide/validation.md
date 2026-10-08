# Validation and comparison with other tools

## Ground truth: synthetic images

{mod}`polarmap.synthetic` generates images and lattices whose displacements and
strains are known exactly:

```python
from polarmap import synthetic

s = synthetic.perovskite_100(shape=(256, 256), a=25.0, displacement=(0.3, -1.2),
                             dose=300, noise_sigma=0.01, seed=1)
seeds = pm.detect_peaks(s.image, 22, threshold_rel=0.2, exclude_border=5)
A = pm.refine_gaussian(s.image, seeds, box=8)
field = pm.measure_cage_displacement(s.image, A, *pm.estimate_lattice_vectors(A))
# compare field with s.displacement (known), e.g. as in tests/test_integration.py
```

Use them to estimate the precision to expect for a given column spacing, column
width and dose, and to test changes of the parameters. The images are sums of
Gaussians (no channelling or aberrations): for accuracy under realistic imaging
conditions, analyse multislice simulations of your structure (e.g. Dr. Probe,
abTEM) in the same way. The PbTiO₃ and PbZrO₃ examples in `examples/data` are
such simulations.

The test suite (`tests/`) is built on this principle; it checks, among other
things, the Gaussian refinement accuracy, the recovery of uniform and
domain-patterned displacements, a non-polar control, exact per-column
deformation gradients for homogeneous strains, shears and rotations, the
insensitivity of Method 2 to a biased reference, and multilayer profiles.

## Column-by-column comparison with VecMap

```python
vecmap = pm.load_vecmap_csv("PTO100-B-site-disp.csv")
cmp = pm.compare_displacement_fields(field, vecmap, sampling, tolerance_px=1.5)
cmp["magnitude_bias_pm"], cmp["vector_rmse_pm"], cmp["median_abs_angle_deg"]
plotting.plot_magnitude_agreement(cmp, labels=("polarmap", "VecMap"))
```

Columns are matched as **mutual nearest neighbours** of the measured target
positions within `tolerance_px`; unmatched columns are excluded and counted.
Both fields must use the same sign convention (measured − reference).

## Comparing strain maps with GPA or peak pairs

For a quantitative comparison, bring both results onto the same points (e.g.
interpolate the GPA map at the column positions, or both onto the same grid)
and use {func}`~polarmap.validation.compare_maps`:

```python
gpa_at_columns = ...                          # GPA value at each column position
pm.compare_maps(tensor.eyy, gpa_at_columns, mask=tensor.valid)
# {'n': ..., 'bias': ..., 'mae': ..., 'rmse': ..., 'std_difference': ..., 'pearson_r': ...}
```

GPA maps have a spatial resolution set by the mask size in reciprocal space,
whereas the per-column values have unit-cell resolution: smooth the per-column
map to a comparable resolution (e.g. average over the same area) before
comparing amplitudes.

## Precision from the data themselves

* {func}`~polarmap.columns.compare_refinements`: Gaussian fit vs centre of mass
  disagreement (a precision proxy for the column positions).
* The spread of per-column strain inside a homogeneous reference region
  (`tensor.eyy[region]`) directly measures the strain noise floor of the
  measurement.
* A **non-polar reference** in the same field of view (e.g. the SrTiO₃
  substrate) gives the displacement noise floor under identical conditions.
