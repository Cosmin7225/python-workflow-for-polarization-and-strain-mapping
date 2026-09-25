# Quick start

This page walks through the two workflows with a minimum of code. Each step is
explained in more detail in the user guide.

## Displacement map of a perovskite [100] image

```python
import numpy as np
import polarmap as pm
from polarmap import plotting

# 1. Image and pixel size (nm/px). Calibrated formats carry the pixel size;
#    for PNG/JPG/TIFF pass it explicitly: pm.load_image("img.png", sampling=0.016)
image, sampling = pm.load_image("examples/data/PTO10040nmHAADF.dm4")

# 2. Reference (A-site) columns: local maxima at the A-A spacing, then a
#    sub-pixel 2-D Gaussian fit. Check the result with plotting.plot_columns.
seeds = pm.detect_peaks(image, min_distance=30, threshold_rel=0.2, exclude_border=5)
A = pm.refine_gaussian(image, seeds, box=10)

# 3. Lattice vectors from the columns themselves.
v1, v2 = pm.estimate_lattice_vectors(A)

# 4. B-column displacement in every complete four-corner cage.
field = pm.measure_cage_displacement(image, A, v1, v2)

# 5. Statistics (magnitudes in pm, circular orientation statistics).
print(pm.format_descriptors(pm.describe_displacements(field, sampling)))

# 6. Figures: arrows coloured by direction, and a colour map of the magnitude.
plotting.plot_displacement_vectors(field, image, sampling=sampling, scalebar_nm="auto")
plotting.plot_displacement_map(field, "magnitude", image=image, sampling=sampling)

# 7. Save the vectors for other programs.
field.to_csv("displacements.csv", sampling=sampling)
```

## Strain maps of a heterostructure

```python
image, sampling = pm.load_image("examples/data/ImgOrigin.mrc")

# Columns and the validated lattice graph shared by both strain methods.
seeds = pm.detect_peaks(image, min_distance=12, threshold_rel=0.1, exclude_border=4)
pos, fit = pm.refine_gaussian(image, seeds, box=4, return_params=True)
pos = pos[fit["success"]]
v1, v2 = pm.estimate_lattice_vectors(pos)
graph = pm.build_lattice_graph(pos, v1, v2)
print(graph.report())

# Method 1: out-of-plane strain relative to an external bulk spacing.
m1 = pm.projection_strain(graph, v2, d0_px=0.3905 / sampling, image_shape=image.shape)

# Method 2: full tensor relative to a reference region of the same image.
reference = pm.fit_reference_lattice(graph, pm.in_rectangle(pos, (100, 1000), (450, 650)))
tensor = pm.tensor_strain(graph, reference, mask=pm.inside_image(pos, image.shape, 20))
print(tensor.report())

plotting.plot_strain_tensor(tensor, sampling=sampling)
profile = pm.binned_profile(pos, tensor.eyy, axis="y", bin_width=v2[1])
plotting.plot_profiles({"eyy (Method 2)": profile}, sampling=sampling)
```

## Where next?

* The [example notebooks](examples.md) run these workflows on the example data
  with all diagnostic figures.
* The [user guide](guide/loading.md) explains every parameter and when to change it.
* The [algorithm overview](algorithm.md) summarises both workflows in a flowchart.
