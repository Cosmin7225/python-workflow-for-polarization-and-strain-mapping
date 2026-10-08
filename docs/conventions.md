# Conventions

| Quantity | Convention |
|---|---|
| Positions | `(x, y)` in pixels; `x` = column index (right), `y` = row index (**down**); arrays of shape `(N, 2)` |
| Pixel size (`sampling`) | nm per pixel |
| Lattice vectors | `v1` is the one closer to the image `x` axis, with `v1[0] > 0`; `v2` has `v2[1] > 0` |
| Displacement `(u, v)` | measured target − reference position, in pixels, image axes (`v > 0` is down) |
| Displacement in pm | `field.magnitude_in(sampling, "pm")` = pixels × sampling × 1000 |
| Orientation, `"cartesian"` (default) | $\theta = \operatorname{atan2}(-v, u)$: 0° right, +90° **up**, counter-clockwise on screen, range (−180°, 180°] |
| Orientation, `"image"` | $\theta = \operatorname{atan2}(v, u)$: 0° right, +90° **down**, clockwise on screen |
| Colour wheel | drawn in the same convention as the colour-coded angles |
| Strain | dimensionless (multiply by 100 for %); positive = tensile |
| Method 1 reference | `d0_px = d0_nm / sampling` |
| Strain-tensor axes | image axes: $x$ right, $y$ down; `in_frame("cartesian")` flips $y$ (changes the sign of $\varepsilon_{xy}$ and $\omega$) |
| Rotation $\omega$ | radians, small-angle; in image axes positive = from $+x$ towards $+y$ (clockwise on screen) |
| Lattice-graph directions | columns of `neighbor_idx` ordered `+v1, -v1, +v2, -v2`; `-1` = no neighbour |
| BFS lattice indices | `(n1, n2)` along `v1`, `v2`; anchor at `(0, 0)`; NaN if not connected |
