# Limitations and good practice

`polarmap` measures positions of intensity maxima and derives displacements and
strain from them. The results are only as reliable as those positions and the
assumptions of each step. The main limitations are listed here together with
ways to check or mitigate them.

## Imaging

* **Apparent vs true column positions.** Intensity maxima need not coincide
  with the projected atomic columns: residual aberrations, specimen mistilt,
  thickness-dependent channelling and overlap of neighbouring columns can shift
  them, more so for light columns and in (A)BF/iDPC signals. Relative
  displacements between columns of the same kind largely cancel such shifts,
  but displacements between different species do not. Validate critical results
  with multislice simulations at the experimental conditions.
* **Choice of signal.** The workflow treats HAADF, iDPC and ABF alike after
  fixing the sign of the contrast, but their transfer and their sensitivity to
  tilt and defocus differ. Use the signal whose contrast reliably locates the
  columns you measure, and keep it fixed within a study.
* **Scan distortions and drift.** In STEM, the image is acquired sequentially,
  so drift and scan noise distort distances, typically differently along the
  fast and slow scan directions. They appear as spurious strain and rotation.
  Correct them before analysis (e.g. non-rigid registration of a frame series,
  or averaging images acquired with orthogonal scan directions); a uniform
  $\omega$ or a difference between $\varepsilon_{xx}$ and $\varepsilon_{yy}$ in an
  unstrained region is a warning sign.
* **Pixel calibration.** Method 1 is directly proportional to the calibration;
  Method 2 is independent of it. Load the original calibrated files, and check
  the calibration on a known lattice when absolute values matter.
* **Image processing.** Denoising (e.g. BM3D) and contrast enhancement (e.g.
  CLAHE) can change the apparent column positions. Compare positions refined on
  the raw and processed images (`compare_displacement_fields`) and inspect the
  difference image (processed − raw) for systematic features before using
  processed images.

## Analysis

* **Reference models are structure specific.** The cage model assumes a
  perovskite-type [100] projection with the target at the cell centre; the pair
  model a target between two reference columns. Other structures need the
  appropriate fractional reference, and the result is a *projected*
  displacement of one sublattice, not a polarization.
* **Sublattice selection.** Detection by local maxima relies on a sensible
  `min_distance` and on the reference sublattice being the brightest. Mixed or
  missing detections produce wrong lattice vectors and many rejected bonds;
  inspect the diagnostics.
* **Reference region (Method 2).** All strain values are relative to the
  chosen region. A region that is not homogeneous (check the fit residuals and
  the pattern of rejected columns) biases every value. Where no unstrained
  region exists, use Method 1 or per-column reference vectors.
* **Interfaces.** Columns at an interface have bonds into both materials; their
  strain is the average of the two spacings, i.e. the interface is located to
  within one unit cell. Chemical intermixing and dislocation cores change the
  column contrast and the local geometry: inspect rejected bonds there. Across
  the whole span of an interface or multilayer the analysis is continuous; see
  [Interfaces and multilayers](guide/strain.md#interfaces-and-multilayers).
* **Outlier rules.** The displacement descriptors use a global rule
  (magnitudes above the median + 6 MAD are excluded from the statistics, not
  from the maps). In strongly heterogeneous samples, e.g. with a minority of
  cells displaced far more than the majority, compare with `n_mad=None`
  (no exclusion) before drawing conclusions from the statistics.
* **Small-strain decomposition.** $\varepsilon = \tfrac12(F + F^\top) - I$ and
  $\omega = \tfrac12(F_{yx} - F_{xy})$ are first-order quantities, accurate
  for strains and rotations of a few per cent. For larger deformations use the
  deformation gradient `tensor.F` directly (e.g. a polar decomposition).
* **Projection.** All quantities are two-dimensional projections through the
  specimen thickness; strain relaxation at the surfaces of a thin lamella and
  inclined interfaces average along the beam direction.
