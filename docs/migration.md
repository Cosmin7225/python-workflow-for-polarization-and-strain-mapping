# From the original notebooks to the package

The first version of this workflow consisted of two notebooks with embedded
code (kept unchanged in `legacy/` for reference). All of that code now lives in
the `polarmap` package, and the notebooks in `examples/` only call it. This page
maps the old names to the new ones.

## Displacement notebook (`Displacement_workflow_complete.ipynb`)

| Notebook | Package | Notes |
|---|---|---|
| `load_stem_image(path, fallback_sampling=...)` → `(s, img, sampling)` | `load_image(path, sampling=...)` → `(image, sampling)` | the HyperSpy signal is no longer returned; `sampling` also overrides a file calibration (with a warning) |
| `as_bright_atoms`, `flatten_background`, `detect_peaks` | same names | `detect_peaks` gained `exclude_border` (the notebook's `margin` step) and merges tied maxima |
| `refine_gaussian`, `refine_com`, `compare_refinements` | same names | `return_params=True` adds `success`; `compare_refinements(image, positions, sampling)` |
| `estimate_lattice_vectors` | same name | fixed averaging for axis-aligned noisy lattices; deterministic order/signs |
| `find_complete_perovskite_cages(..., return_corner_indices=True)` | `find_complete_perovskite_cages(...)` | always returns `(centres, corner_indices)` |
| `remove_columns_gaussian` | same name | |
| `measure_polarization(...)` → `x, y, u, v` | `measure_cage_displacement(...)` → `DisplacementField` | `field.x, field.y, field.u, field.v`; `sign=` replaces `POLARISATION_SIGN` |
| `predict_second_sublattice` | same name | |
| `polarization_from_sublattices` | `displacement_from_sublattices` | |
| `displacement_from_cage`, `displacement_from_offset` | same names | return a `DisplacementField` |
| `split_by_intensity`, `split_columns_by_fitted_amplitude` | `split_by_amplitude` → `IntensitySplit` | `.bright`, `.dim`, `.threshold` (refined positions) |
| `estimate_pair_vector`, `find_complete_reference_pairs` | same names | pairs: always returns `(ideal_xy, pair_indices)` |
| `local_residual_maxima` | same name | |
| `measure_pair_centered_displacement` → dict | `measure_pair_displacement` → `DisplacementField` | extras in `field.per_site` and `field.diagnostics["residual"]` |
| `displacement_to_polarization` | same name | |
| statistics cell (MAD rule, circular statistics) | `describe_displacements`, `format_descriptors` | orientations in the Cartesian convention by default |
| `plot_polarisation_angle` / `_magnitude` / `_overlay` | `plotting.plot_displacement_vectors(color_by="angle" / "magnitude" / None)` | colour wheel follows the angle convention |
| — | `plotting.plot_displacement_map` | colour maps without arrows |
| `show_dimmed_image`, `add_colorwheel` | `plotting.show_image(..., dim=True)`, `plotting.add_colorwheel` | |
| `load_vecmap_displacements` (pandas) | `load_vecmap_csv` → `DisplacementField` | no pandas dependency |
| `mutual_nearest_matches`, `angular_difference_deg`, metrics cells | `mutual_nearest_matches`, `angular_difference`, `compare_displacement_fields` | |

## Strain notebook (`Strain_map_python_routine.ipynb`)

| Notebook | Package | Notes |
|---|---|---|
| Atomap `get_atom_positions` / `Sublattice` / `construct_zone_axes` | `detect_peaks` + `refine_gaussian` + `estimate_lattice_vectors` | Atomap positions can still be passed in as an `(N, 2)` array |
| `za0`, `za1` | `v1`, `v2` | |
| `build_lattice_graph(pos, [za0, za1])` → arrays | `build_lattice_graph(pos, v1, v2)` → `LatticeGraph` | `neighbor_idx`, `neighbor_vec`, `bad_edges`, `clean_idx`; new perpendicular limit |
| `find_bad_edges_by_closure`, `bfs_lattice_indices(idx, anchor, n)` | same names; `bfs_lattice_indices(idx, anchor)` | |
| `plot_atom_neighbor_directions` | `plotting.plot_neighbor_diagnostic`, `LatticeGraph.describe_column` | |
| Method 1 cell (`eps_c`, `eps_clean`) | `projection_strain(graph, direction, d0_px, image_shape=...)` | local outlier test by default; `status` codes explain exclusions |
| Step 1 reference rectangle (`REF_X_MIN`, ...) | `in_rectangle(pos, x_range, y_range)` | |
| Step 2 Stage A/B (`za0_fit`, `za1_fit`, `origin_fit`) | `fit_reference_lattice(graph, mask)` → `ReferenceLattice` | `.a`, `.b`, `.origin`, `.used_mask` |
| Step 3 `u_disp_m2` | `rigid_lattice_displacement(graph, reference)` | diagnostic only |
| Step 4 `F_tensor`, `exx_atom`, ... | `tensor_strain(graph, reference, mask=...)` → `TensorStrain` | `.F`, `.exx`, `.eyy`, `.exy`, `.omega` |
| `griddata` heat maps (Step 5) | `interpolate_to_grid`, `plotting.plot_strain_tensor(kind="grid")` | |
| Steps 6-7 comparison and line profile | `plotting.plot_strain_comparison`, `line_profile`, `binned_profile` | |
| `draw_planes` | `plotting.plot_lattice_planes` | |
| neighbour displacement cloud | `plotting.plot_neighbor_cloud` | |
