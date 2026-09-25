# Figures

All plotting functions live in {mod}`polarmap.plotting`. Each draws into a
Matplotlib axes (a new figure is created unless `ax=` is given) and returns the
figure and axes, so panels can be combined and saved freely:

```python
import matplotlib.pyplot as plt
from polarmap import plotting

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plotting.plot_displacement_vectors(field, image, ax=axes[0], sampling=sampling,
                                   scalebar_nm="auto")
plotting.plot_displacement_map(field, "magnitude", ax=axes[1], image=image,
                               sampling=sampling, colorbar_location="bottom")
fig.savefig("figure1.tiff", dpi=300, bbox_inches="tight")
```

## Arrows or colour maps?

Arrows show magnitude and direction at once, which suits figures where both
matter. When only one quantity is of interest, a colour map is easier to read:

| Question | Figure |
|---|---|
| where and in which direction? | `plot_displacement_vectors(field, image, color_by="angle")` |
| how large, and where? | `plot_displacement_map(field, "magnitude")` |
| which direction, and where (domains)? | `plot_displacement_map(field, "angle")` |
| signed component, e.g. up vs down | `plot_displacement_map(field, "v")` |

`plot_displacement_map` offers three renderings: `kind="tiles"` fills each unit
cell with its value (a parallelogram spanned by the lattice vectors),
`kind="interpolated"` interpolates linearly between cells, and
`kind="scatter"` draws one marker per cell.

## Orientation colour coding

Orientations use a cyclic colour map (colorcet's perceptually uniform
`cet_colorwheel` if installed, otherwise `hsv`) with a colour wheel drawn in the
**same angle convention** as the data (`convention="cartesian"` by default:
+90° is at the top of the wheel and points up in the image).

## Scale bars and colour bars

* `scalebar_nm="auto"` adds a scale bar of a round length (about 20 % of the
  width); give a number to fix the length.
* Colour bars are appended to the image axes, so they always match its height
  (or width): `colorbar_location="right"` or `"bottom"`. Horizontal bars under
  each panel keep multi-panel strain figures compact.
* {func}`~polarmap.plotting._common.add_scalebar` and
  {func}`~polarmap.plotting._common.add_colorbar` can be used on any figure.

## Strain figures

| Figure | Function |
|---|---|
| one component, per column | `plot_strain_scatter(pos, values)` |
| one component, triangulated | `plot_strain_triangulation(pos, values)` |
| all tensor components | `plot_strain_tensor(tensor, kind="scatter" or "grid")` |
| two maps with shared limits | `plot_strain_comparison(pos, a, b)` |
| layer profiles | `plot_profiles({"label": binned_profile(...)})` |

Colour limits are symmetric around zero at the 98th percentile of |value| by
default (diverging colour map: red tensile, blue compressive); pass explicit
limits for figures that must be compared.

## Diagnostics

Before interpreting a map, check each step:

| Check | Function |
|---|---|
| one marker per intended column | `plot_columns(image, positions)` |
| lattice vectors follow the atomic planes | `plot_lattice_planes(image, positions, v1)` |
| first-shell vectors form tight clusters | `plot_neighbor_cloud(positions)` |
| complete cages and fitted B columns | `plot_cages(field, A)` |
| complete pairs and fitted weak columns | `plot_pairs(field, reference)` |
| bright/dim split | `plot_intensity_split(image, split)` |
| bonds of a suspicious column | `plot_neighbor_diagnostic(graph, index, image=image)` |
| reference columns used / rejected | `plot_reference_region(image, graph, reference)` |
