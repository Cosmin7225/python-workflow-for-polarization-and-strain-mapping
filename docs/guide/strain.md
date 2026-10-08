# Strain mapping

Strain is measured in real space from the refined column positions. Both
methods share one ingredient, the **validated lattice graph**, and differ in the
reference against which the lattice is compared.

|  | Method 1: projection strain | Method 2: tensor strain |
|---|---|---|
| Reference | external spacing $d_0$ (e.g. bulk) | region of the same image |
| Output | one component, along a chosen direction | $\varepsilon_{xx}$, $\varepsilon_{yy}$, $\varepsilon_{xy}$ and rotation $\omega$ |
| Unstrained region needed | no | yes |
| Pixel-calibration error | shifts every value | cancels |
| Function | {func}`~polarmap.strain.projection_strain` | {func}`~polarmap.strain.fit_reference_lattice` + {func}`~polarmap.strain.tensor_strain` |

## The validated lattice graph

```python
v1, v2 = pm.estimate_lattice_vectors(pos)
graph = pm.build_lattice_graph(pos, v1, v2)
print(graph.report())
```

For every column and each of the four bond directions $\pm\mathbf v_1$,
$\pm\mathbf v_2$ (unit vector $\hat{\mathbf u}$, length $L$),
{func}`~polarmap.lattice.build_lattice_graph` selects the neighbour as follows:

1. candidates are columns whose projection on $\hat{\mathbf u}$ lies between
   $0.5L$ and $1.5L$ and whose perpendicular distance to the bond line is at
   most $0.5L$ (halfway to the adjacent row);
2. among them, the one with the **smallest perpendicular offset** is chosen.

Choosing the most in-line candidate, rather than the one with the smallest
projected distance, avoids picking a column of the adjacent row under local
shear or scan jitter. The perpendicular limit prevents a diagonal column from
being accepted when the true neighbour is missing (border, vacancy, undetected
column).

**Loop closure.** Every bond is then validated on the elementary cells it
belongs to: from any column, walking $+\mathbf v_1$ then $+\mathbf v_2$ must
end on the same column as walking $+\mathbf v_2$ then $+\mathbf v_1$ (and
likewise for the other three sign combinations). When the two paths end on
different columns, at least one of the four bonds of that cell is wrong, and all
four are rejected ({func}`~polarmap.lattice.find_bad_edges_by_closure`). This is the same
idea as the residue test of two-dimensional phase unwrapping: inconsistent loops
are removed before anything is propagated through them. Only validated bonds
({attr}`~polarmap.lattice.LatticeGraph.clean_idx`) are used by either method.

To see why a particular column has no value, inspect its bonds:

```python
print(graph.describe_column(1234))
plotting.plot_neighbor_diagnostic(graph, 1234, image=image)
```

## Method 1: strain along one direction, against an external reference

```python
m1 = pm.projection_strain(graph, v2, d0_px=0.3905 / sampling,
                          image_shape=image.shape, edge=20)
print(m1.report())
```

With $\hat{\mathbf u}$ the chosen direction, the local spacing at column $i$ is

$$
d_i = \tfrac12\left(\mathbf b_{i,+}\cdot\hat{\mathbf u}
      - \mathbf b_{i,-}\cdot\hat{\mathbf u}\right),
$$

the symmetric average of its validated bonds $\mathbf b_{i,\pm}$ along
$\pm\hat{\mathbf u}$ (or the one available bond), and the engineering strain is

$$
\varepsilon_i = \frac{d_i - d_0}{d_0}.
$$

* $d_0$ is given in pixels: `d0_nm / sampling`. It inherits any error of the
  calibration (see [Loading](loading.md)).
* `d0_px` may be an array with one value per column, to refer each layer of a
  heterostructure to its own bulk spacing (see below).
* Excluded columns are counted by reason in `m1.status` / `m1.report()`: no
  neighbour, bonds rejected by loop closure, edge margin, robust outlier.
* **Outliers** are columns deviating from the median of their 8 nearest
  neighbours by more than 5 robust standard deviations of these deviations
  ({func}`~polarmap.statistics.local_outlier_mask`). A local rule is essential
  for heterostructures: a global median/MAD rule treats a strained layer that
  covers less than about half of the columns as a whole as outliers
  (`outlier_scope="global"` reproduces that behaviour for comparison).

## Method 2: strain and rotation tensor, against an internal reference

### Step 1: reference lattice

Choose a region you know or assume to be unstrained (substrate, a thick layer
away from interfaces) and fit a reference lattice to it:

```python
region = pm.in_rectangle(pos, x_range=(100, 1000), y_range=(450, 650))  # pixels
reference = pm.fit_reference_lattice(graph, region)
print(reference.report(sampling))
plotting.plot_reference_region(image, graph, reference,
                               x_range=(100, 1000), y_range=(450, 650))
```

1. **Indexing by breadth-first search.** Starting from the reference column
   nearest to the centre of the region (index $(0, 0)$), every column reached
   through a validated bond receives the index of the column it was reached
   from, plus or minus one along the bond direction. Because each index is
   defined relative to an already indexed neighbour, a local ambiguity cannot
   flip the indices of the whole region beyond it, which is what happens when
   indices are obtained by rounding $(\mathbf r - \mathbf r_0)/\mathbf v$ from a
   distant origin once the accumulated drift exceeds half a cell.
2. **Least-squares fit.** The model
   $\mathbf r_i = \mathbf r_0 + n_{1,i}\,\mathbf a + n_{2,i}\,\mathbf b$ is fitted
   to the reference columns; the 20 % with the largest residuals are rejected
   and the model is refitted. The report lists the residuals before and after,
   and the fitted vectors in nm.

A good reference region gives sub-pixel residuals and fitted vectors close to
the expected bulk spacings. A large residual or a strongly clustered pattern of
rejected columns indicates that the region is not homogeneous.

### Step 2: per-column deformation gradient

```python
tensor = pm.tensor_strain(graph, reference,
                          mask=pm.inside_image(pos, image.shape, margin=20))
print(tensor.report())
```

For column $i$, each validated bond $d$ gives a measured vector
$\mathbf m_d$ and the corresponding reference vector
$\mathbf e_d \in \{\pm\mathbf a, \pm\mathbf b\}$. The local deformation gradient
is the least-squares solution of

$$
\mathbf m_d \approx F_i\,\mathbf e_d
\quad\Longrightarrow\quad
F_i = \Big(\sum_d \mathbf m_d \mathbf e_d^{\top}\Big)
      \Big(\sum_d \mathbf e_d \mathbf e_d^{\top}\Big)^{-1},
$$

which requires at least one validated bond along each lattice direction. Strain
and rigid-body rotation are its symmetric and antisymmetric parts
(small-strain decomposition):

$$
\varepsilon_i = \tfrac12\left(F_i + F_i^{\top}\right) - I,
\qquad
\omega_i = \tfrac12\left(F_{i,yx} - F_{i,xy}\right).
$$

**Why per column?** $F_i$ uses only the bonds of column $i$. An error in one
column (a poorly fitted column, a residual wrong bond) affects that column
alone, and a small bias of the reference vectors produces a *uniform* offset
instead of an error growing with distance from the reference region. The
alternative, interpolating the displacements from a rigidly extrapolated
lattice and differentiating them, amplifies both problems: any mismatch of the
reference vectors makes the displacement grow linearly with distance, and
finite differences turn single bad columns into spatially extended artefacts.
{func}`~polarmap.strain.rigid_lattice_displacement` is kept only as a
diagnostic of that drift.

Strain is a ratio of pixel distances, so Method 2 does not depend on the pixel
calibration; `sampling` only sets the axis units of the figures.

### Axes, signs and rotation

Components are expressed in image axes ($x$ right, $y$ down). In these axes a
positive $\omega$ is a rotation from $+x$ towards $+y$, i.e. clockwise on the
screen. $\varepsilon_{xx}$ and $\varepsilon_{yy}$ do not depend on the direction
of the $y$ axis, whereas $\varepsilon_{xy}$ and $\omega$ change sign in a y-up
frame:

```python
tensor.in_frame("cartesian")                  # y up: exy and omega change sign
tensor.in_frame("rotated", angle_deg=12.0)    # axes rotated, e.g. along the lattice
```

If the growth direction is not aligned with the image $y$ axis, rotate the
tensor so that $\varepsilon_{yy}$ is the out-of-plane component.

**Reading the components.** $\varepsilon_{xx}$ and $\varepsilon_{yy}$ are
changes of the lattice spacings along $x$ and $y$. $\varepsilon_{xy}$ is shear:
a change of the angle between the two lattice directions, as produced by a
monoclinic distortion or a strain gradient near a dislocation or domain wall.
$\omega$ is a rigid rotation of the local lattice without any change of bond
lengths or angles. A uniform $\omega$ over the whole image usually reflects a
small misalignment of the reference; a localised $\omega$, for example at a
boundary, can be structural, but check first that the columns there have all
four bonds (reduced bond statistics near borders make $F_i$ noisier).

### Visualising

```python
plotting.plot_strain_tensor(tensor, sampling=sampling)               # per column
plotting.plot_strain_tensor(tensor, sampling=sampling, kind="grid")  # interpolated
```

The per-column (scatter) view is the measurement. The grid view interpolates
the per-column values linearly for display only; compare it with the scatter
view before interpreting a feature.

## Interfaces and multilayers

The per-column formulation analyses the whole span of an interface or a
multilayer in a single map. Three complementary options are available:

1. **One reference for the whole image** (e.g. the substrate):
   `pm.tensor_strain(graph, reference)`. Every layer is measured relative to the
   same lattice, which is what is needed to follow lattice-parameter changes
   across interfaces (misfit, relaxation).
2. **Each layer relative to its own bulk lattice**: pass one pair of reference
   vectors per column, e.g. from layer labels,

   ```python
   ref = np.empty((graph.n_atoms, 2, 2))
   ref[:, 0] = [a_px, 0.0]                              # in-plane, coherent
   ref[:, 1] = np.where(in_layer_B, c_B_px, c_A_px)[:, None] * [0.0, 1.0]
   tensor = pm.tensor_strain(graph, ref)
   ```

   Method 1 accepts a per-column `d0_px` in the same way.
3. **Layer profiles**: average the per-column values in bands parallel to the
   interfaces, with their spread,

   ```python
   profile = pm.binned_profile(pos, tensor.eyy, axis="y", bin_width=v2[1])
   plotting.plot_profiles({"eyy": profile}, sampling=sampling)
   ```

Columns *at* an interface have bonds into both layers; their value is the
symmetric average of the two spacings, which locates the interface to within
one unit cell. The graph requires a continuous lattice: an incoherent interface
with misfit dislocations is indexed correctly as long as the bonds across it
remain within the matching window, but the dislocation cores themselves will
show rejected bonds (inspect them with `describe_column`).
