# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [0.1.0] - unreleased

First release as an installable, tested Python package. The code of the two
original notebooks (now in `legacy/`) was reorganised into the `polarmap`
package; the notebooks in `examples/` call the package instead of embedding
the code.

### Added
- `polarmap` package (`src/` layout) with thematic modules: `io`, `columns`,
  `lattice`, `displacement`, `strain`, `statistics`, `validation`,
  `synthetic` and `plotting`.
- Installation with `pip` (`pyproject.toml`), optional extras `io` (HyperSpy),
  `colors` (colorcet), `all`, `test`, `docs`; `environment.yml` for conda.
- Test suite (pytest) with synthetic ground truth, regression tests pinned to
  the published example results, and continuous integration on Linux, macOS
  and Windows, including the oldest supported NumPy/SciPy/Matplotlib and a
  weekly run against the newest releases.
- Documentation website (MkDocs) with a user guide, an algorithm flowchart,
  a limitations section and the API reference.
- Built-in MRC reader that tolerates Thermo Fisher files whose header
  declares an extended header that is missing (the case of
  `examples/data/ImgOrigin.mrc`, previously misnamed `.dm4`).
- Result objects (`DisplacementField`, `LatticeGraph`, `ProjectionStrain`,
  `ReferenceLattice`, `TensorStrain`) with `summary()`/`report()` methods,
  CSV export and explicit status codes for excluded columns.
- Colour-map displacement figures without arrows (tiles, interpolated or
  scatter), scale bars, colour bars sized to the image (optionally
  horizontal), colour wheels drawn in the same angle convention as the data.
- Per-column reference lattices for Method 2 and per-column `d0` for Method 1,
  so each layer of a heterostructure can be referred to its own bulk lattice.
- `binned_profile` (layer profiles from per-column values), `compare_maps`
  (bias/MAE/RMSE/Pearson r between maps, e.g. against GPA), VecMap CSV reader
  without pandas, `local_outlier_mask`, `TensorStrain.in_frame` (Cartesian or
  rotated axes).
- Synthetic perovskite [100]/[110] images and strained lattices with exact
  ground truth (`polarmap.synthetic`).

### Changed
- Column refinement uses an analytic Jacobian for the 2-D Gaussian fit
  (about 2x faster, identical results to 1e-6 px on the examples);
  `refine_gaussian` reports a per-column `success` flag.
- Orientations default to the Cartesian convention (0 deg right, +90 deg up)
  for statistics *and* colour coding, as described in the manuscript; the
  image convention is available with `convention="image"`.
- Renamed: `measure_polarization` -> `measure_cage_displacement`,
  `measure_pair_centered_displacement` -> `measure_pair_displacement`,
  `polarization_from_sublattices` -> `displacement_from_sublattices`,
  `split_columns_by_fitted_amplitude`/`split_by_intensity` ->
  `split_by_amplitude`, `load_stem_image` -> `load_image`.
- The strain workflow no longer requires Atomap: columns are detected and
  refined with the same routines as the displacement workflow (positions
  from any other program can still be passed in).
- Dependencies reduced to NumPy, SciPy and Matplotlib; HyperSpy is optional.

### Fixed
- `estimate_lattice_vectors`: for a lattice aligned with an image axis, noise
  split the bond vectors between `+v` and `-v` before averaging, which could
  collapse a lattice vector to ~1 px. Vectors are now oriented along their
  axis before averaging, and returned in a deterministic order.
- `build_lattice_graph`: a diagonal column could be accepted as a neighbour
  when the true neighbour was missing (image border, vacancy), which the
  loop-closure test then rejected together with valid bonds. Candidates are now
  limited to half a lattice spacing from the bond line (`max_perp_frac`); on
  `ImgOrigin.mrc` this removes all 161 rejected bonds and gives a strain tensor
  to 7929 instead of 7893 of 7935 columns.
- Method 1 outlier test: the global median/MAD rule flagged a strained layer
  covering less than about half of the columns as a whole. The default is now a
  local rule (deviation from the median of the 8 nearest columns); the global
  rule remains available with `outlier_scope="global"`.
- `displacement_angle` returned -180 instead of +180 deg for vectors along -x
  (negative zero); `angular_difference` now always lies in [-180, 180).
- Measurement windows and the residual image used for the diagnostic figure are
  now the ones actually used in the measurement.
