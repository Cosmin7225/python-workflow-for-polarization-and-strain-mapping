# Displacement mapping

A displacement map is always *the measured position of a target (polar) column
minus the position it would occupy in the centrosymmetric structure*. Column
localisation is generic; what depends on the crystal structure and projection
is how that **reference position** is constructed. `polarmap` builds it from the
*measured* neighbouring reference columns, never from a lattice extrapolated
over the whole image, so the reference follows local lattice distortions.

| Projection | Reference columns | Ideal target position | Function |
|---|---|---|---|
| perovskite [100] | 4 A columns forming a complete cage | centroid of the four measured corners | {func}`~polarmap.displacement.measure_cage_displacement` |
| perovskite [110] | 2 A columns along the projected polar axis | $(1-f)\,\mathbf r_0 + f\,\mathbf r_1$, $f = 1/2$ | {func}`~polarmap.displacement.measure_pair_displacement` |
| dumbbell / wurtzite | 1 reference column | reference + fixed basis offset | {func}`~polarmap.displacement.displacement_from_offset` |
| any, positions known | both sublattices measured | cage centroid or lattice site | {func}`~polarmap.displacement.displacement_from_cage`, {func}`~polarmap.displacement.displacement_from_sublattices` |

All functions return a {class}`~polarmap.displacement.DisplacementField` holding the
reference positions `(x, y)` and displacements `(u, v)` in pixels.

## Perovskite [100]: complete four-corner cages

```python
field = pm.measure_cage_displacement(image, A, v1, v2, image_mode="HAADF")
```

1. **Complete cages.** For every A column taken as anchor $\mathbf A_0$, the
   expected corners are $\mathbf A_0$, $\mathbf A_0 + \mathbf v_1$,
   $\mathbf A_0 + \mathbf v_2$ and $\mathbf A_0 + \mathbf v_1 + \mathbf v_2$.
   The cage is accepted only if a measured A column lies within `cage_tolerance`
   (default 0.28 of the shorter spacing) of each expected corner, the four
   columns are distinct, and their centroid is within the same tolerance of
   $\mathbf A_0 + (\mathbf v_1 + \mathbf v_2)/2$. The reference is the
   centroid of the **four measured** corners. Cells at the image border, next
   to vacancies or missed columns are therefore rejected rather than given a
   false reference.
2. **Reference subtraction.** A Gaussian is fitted to every A column and
   subtracted (window half-width `max(4, 0.45 cell)`). On real, blurred images
   the tails of the bright A columns extend into the cell centre and would pull
   a fit of the weak B column towards them.
3. **Target refinement.** The B column is fitted with a 2-D Gaussian on the
   residual image, starting from the cage centroid (window half-width
   `max(3, 0.30 cell)`).
4. **Quality control.** A vector is kept if the fit succeeded, the displacement
   is below `max_displacement` (0.34 cell) and the fitted amplitude exceeds
   `min_amplitude` (0.30) times the median amplitude.

Here `cell` is the mean of $|\mathbf v_1|$ and $|\mathbf v_2|$. For strongly
strained data, increase `cage_tolerance` slightly (0.30–0.35 of the spacing,
always below 0.5).

The residual image and the corner indices are kept in the result, so the
diagnostic figure shows exactly what was measured:

```python
plotting.plot_cages(field, A)          # cages, centroids and fitted B columns
```

## Perovskite [110] and other two-column references

In the [110] projection the B-containing column sits between two bright A
columns along the projected polar axis. Detect all columns, split them by
intensity, estimate the pair vector, then measure:

```python
split = pm.split_by_amplitude(image, pm.detect_peaks(image, 8), box=3, edge=10)
pair_vector = pm.estimate_pair_vector(split.bright, axis_hint=(0, 1))
field = pm.measure_pair_displacement(image, split.bright, pair_vector)
```

* A pair is **complete** only if a measured reference column lies at both ends
  of the pair vector; the ideal position is interpolated between the two
  *measured* endpoints.
* `axis_hint` is a rough direction in image coordinates (`x` right, `y` down)
  from one reference column to the next along the polar axis.
* `reference_fraction` $f$ places the ideal position along the pair: $f = 1/2$
  is the midpoint (perovskite [110]). An asymmetric internal coordinate, such as
  $u = 3/8$ for the cation-to-anion spacing of wurtzite along $c$, is set the
  same way; the sign of `axis_hint` then decides which end is $\mathbf r_0$.
* The weak column is seeded at the brightest residual pixel near the ideal
  position (after subtracting the reference columns) and refined with a
  Gaussian fit; fits with an ellipticity above 3, a displacement above 0.30 of
  the pair spacing or an amplitude below 0.25 of the median are rejected.

## Other structures

If you know where the polar column sits in the projected cell, expressed as a
fraction between columns you can see, you can map its displacement: the pair
model covers any two-column reference, and
{func}`~polarmap.displacement.displacement_from_offset` covers a fixed offset from one
reference column (`ideal_offset_px = (0, u_ref * c / sampling)` for a wurtzite
projection with $c$ along $+y$). Structure- and imaging-specific validation
(simulation) is essential for light columns.

## Sign and orientation conventions

* $(u, v)$ = measured target − reference, in **image axes** ($v > 0$ is down).
  The arrow therefore points along the cation shift. Use `sign=-1` (or
  `field.with_sign(-1)`) for the opposite convention.
* Orientations use the **Cartesian** convention by default:
  $\theta = \operatorname{atan2}(-v, u)$, so 0° points right and +90° points
  up; `convention="image"` gives $\operatorname{atan2}(v, u)$. Statistics,
  colour coding and colour wheels always use the same convention.

See [Conventions](../conventions.md) for the full list.

## Statistics

```python
desc = pm.describe_displacements(field, sampling)        # dict
print(pm.format_descriptors(desc))
plotting.plot_displacement_statistics(field, sampling)   # histograms and rose
```

* Magnitudes above `median + 6 MAD` (raw median absolute deviation, upper side
  only) are excluded before the statistics; the number excluded is reported.
* Orientations are summarised with **circular statistics**: the direction of the
  mean resultant vector (circular mean), its length $R$ (1 = all vectors
  parallel, 0 = isotropic) and the circular standard deviation
  $\sqrt{-2 \ln R}$. An arithmetic mean of angles would be wrong near ±180°.

## From displacement to polarization

```python
P = pm.displacement_to_polarization(desc["median_pm"], z_star=7.1,
                                    cell_volume_nm3=0.402 * 0.402 * 0.413)
```

$P = e Z^* u / \Omega$ is a semi-quantitative, single-sublattice estimate
(µC/cm²). A rigorous polarization sums $Z^*_\kappa \mathbf u_\kappa$ over all
ions, including oxygen, with anisotropic Born-charge tensors. Report the
measured displacement (pm) as the primary quantity.

## Validation

The test suite checks, on synthetic images with known displacements:

* noise-free accuracy better than 0.02 px, and an RMS error below 0.15 px
  without systematic bias at a dose typical of experiments;
* a **non-polar control**: a centrosymmetric structure gives a mean
  displacement below 0.03 px;
* opposite domains are resolved;
* on the simulated PbTiO₃ [100] example, 64 complete cages and a median Ti
  displacement of 20.545 pm; the column-by-column comparison with VecMap
  (median 20.715 pm) gives a magnitude bias of −0.17 pm and a median angular
  difference of 0.04° (see [Validation](validation.md)).
