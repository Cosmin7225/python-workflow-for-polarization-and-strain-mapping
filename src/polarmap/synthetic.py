"""Synthetic images and lattices with exactly known ground truth.

These generators are used by the test suite and the tutorials to verify that
the analysis recovers known displacements and strains, and can be used to
estimate the precision of a given acquisition setting (column separation,
column width, noise level) before or alongside real data.

The images are sums of isotropic 2-D Gaussians on a constant background,
optionally with Poisson (shot) and Gaussian noise. They are *not* image
simulations: they contain no channelling, probe or aberration effects (use a
multislice code such as Dr. Probe or abTEM for those).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "render_columns",
    "add_noise",
    "SyntheticImage",
    "perovskite_100",
    "perovskite_110",
    "SyntheticLattice",
    "lattice_points",
    "strained_lattice",
    "superlattice",
]


def render_columns(positions, amplitudes, sigmas, shape, background=0.0):
    """Render atomic columns as isotropic 2-D Gaussians.

    Parameters
    ----------
    positions : array_like
        ``(N, 2)`` column centres ``(x, y)`` in pixels (may lie outside the
        image; only the part inside is drawn).
    amplitudes, sigmas : float or array_like
        Peak heights and standard deviations (px), scalar or ``(N,)``.
    shape : (int, int)
        Image shape ``(height, width)``.
    background : float, default 0
        Constant background.

    Returns
    -------
    numpy.ndarray
        ``float64`` image.
    """
    pos = np.asarray(positions, dtype=float).reshape(-1, 2)
    amps = np.broadcast_to(np.asarray(amplitudes, dtype=float), (len(pos),))
    sigs = np.broadcast_to(np.asarray(sigmas, dtype=float), (len(pos),))
    h, w = shape
    image = np.full((h, w), float(background))
    for (x, y), amp, sig in zip(pos, amps, sigs, strict=True):
        r = int(np.ceil(5 * sig))
        x0, x1 = max(int(np.floor(x)) - r, 0), min(int(np.floor(x)) + r + 2, w)
        y0, y1 = max(int(np.floor(y)) - r, 0), min(int(np.floor(y)) + r + 2, h)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        image[y0:y1, x0:x1] += amp * np.exp(
            -((xx - x) ** 2 + (yy - y) ** 2) / (2 * sig ** 2))
    return image


def add_noise(image, dose=None, gaussian_sigma=0.0, seed=None):
    """Add shot noise and/or Gaussian noise.

    Parameters
    ----------
    image : array_like
        Noise-free image with non-negative values.
    dose : float, optional
        Counts per unit intensity; the image is replaced by
        ``Poisson(dose * image) / dose``. Smaller doses are noisier.
    gaussian_sigma : float, default 0
        Standard deviation of additive Gaussian noise.
    seed : int or numpy.random.Generator, optional
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    out = np.asarray(image, dtype=float)
    if dose is not None:
        out = rng.poisson(np.clip(out, 0, None) * dose) / float(dose)
    if gaussian_sigma:
        out = out + rng.normal(0.0, gaussian_sigma, out.shape)
    return out


def _grid(shape, v1, v2, origin, pad):
    """All lattice points origin + i v1 + j v2 within the padded image."""
    h, w = shape
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    corners = np.array([[-pad, -pad], [w + pad, -pad], [-pad, h + pad],
                        [w + pad, h + pad]]) - np.asarray(origin, dtype=float)
    ij = corners @ np.linalg.inv(np.vstack([v1, v2]))
    i = np.arange(np.floor(ij[:, 0].min()) - 1, np.ceil(ij[:, 0].max()) + 2)
    j = np.arange(np.floor(ij[:, 1].min()) - 1, np.ceil(ij[:, 1].max()) + 2)
    ii, jj = np.meshgrid(i, j)
    pts = origin + ii.ravel()[:, None] * v1 + jj.ravel()[:, None] * v2
    keep = ((pts[:, 0] >= -pad) & (pts[:, 0] < w + pad)
            & (pts[:, 1] >= -pad) & (pts[:, 1] < h + pad))
    return pts[keep], np.column_stack([ii.ravel(), jj.ravel()])[keep]


def _inside(points, shape):
    h, w = shape
    return ((points[:, 0] >= 0) & (points[:, 0] <= w - 1)
            & (points[:, 1] >= 0) & (points[:, 1] <= h - 1))


@dataclass
class SyntheticImage:
    """A synthetic two-sublattice image with its ground truth.

    Attributes
    ----------
    image : numpy.ndarray
        The rendered (and possibly noisy) image.
    sampling : float
        Nominal pixel size in nm/px.
    reference : numpy.ndarray
        ``(N, 2)`` true positions of the bright reference columns inside the
        image.
    target : numpy.ndarray
        ``(M, 2)`` true positions of the displaced target columns inside the
        image.
    target_ideal : numpy.ndarray
        ``(M, 2)`` centrosymmetric positions of those target columns.
    v1, v2 : numpy.ndarray
        Lattice vectors of the reference sublattice.
    info : dict
        Generation parameters.
    """

    image: np.ndarray
    sampling: float
    reference: np.ndarray
    target: np.ndarray
    target_ideal: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    info: dict = field(default_factory=dict)

    @property
    def displacement(self):
        """``(M, 2)`` true displacement of each target column (px)."""
        return self.target - self.target_ideal


def _displacement_values(displacement, ideal):
    if callable(displacement):
        d = np.asarray(displacement(ideal), dtype=float)
    else:
        d = np.broadcast_to(np.asarray(displacement, dtype=float), ideal.shape)
    if d.shape != ideal.shape:
        raise ValueError("displacement must give one (dx, dy) per target column")
    return np.array(d)


def perovskite_100(shape=(256, 256), a=25.0, c=None, displacement=(0.0, -1.3),
                   amplitude_a=1.0, amplitude_b=0.35, sigma_a=2.8, sigma_b=2.4,
                   background=0.05, rotation_deg=0.0, origin=None, dose=None,
                   noise_sigma=0.0, sampling=0.016, seed=None):
    """Synthetic perovskite [100]-type image: A cages with a central B column.

    Parameters
    ----------
    shape : (int, int), default (256, 256)
    a, c : float
        A-A spacings (px) along the rotated ``x`` and ``y`` axes; ``c``
        defaults to ``a``.
    displacement : (float, float) or callable, default (0, -1.3)
        B-column displacement from the cage centre in pixels (image axes; the
        default points up). A callable receives the ``(M, 2)`` ideal positions
        and returns ``(M, 2)`` displacements, e.g. to create domains.
    amplitude_a, amplitude_b : float
        Peak heights of the A and B columns.
    sigma_a, sigma_b : float
        Gaussian widths (px).
    background : float, default 0.05
    rotation_deg : float, default 0
        Rotation of the lattice (from ``+x`` towards ``+y``).
    origin : (float, float), optional
        Position of one A column; default ``(0.6 a, 0.55 c)``.
    dose, noise_sigma, seed
        Noise, see :func:`add_noise` (no noise by default).
    sampling : float, default 0.016
        Nominal pixel size reported with the image (nm/px).

    Returns
    -------
    SyntheticImage
    """
    c = a if c is None else c
    t = np.deg2rad(rotation_deg)
    v1 = a * np.array([np.cos(t), np.sin(t)])
    v2 = c * np.array([-np.sin(t), np.cos(t)])
    origin = np.array([0.6 * a, 0.55 * c]) if origin is None else np.asarray(origin, float)
    pad = 3 * max(a, c)
    A, _ = _grid(shape, v1, v2, origin, pad)
    ideal = A + 0.5 * (v1 + v2)
    B = ideal + _displacement_values(displacement, ideal)

    image = (render_columns(A, amplitude_a, sigma_a, shape, background)
             + render_columns(B, amplitude_b, sigma_b, shape, 0.0))
    if dose is not None or noise_sigma:
        image = add_noise(image, dose=dose, gaussian_sigma=noise_sigma, seed=seed)
    in_a, in_b = _inside(A, shape), _inside(B, shape)
    return SyntheticImage(image, sampling, A[in_a], B[in_b], ideal[in_b], v1, v2,
                          info=dict(kind="perovskite_100", a=a, c=c,
                                    rotation_deg=rotation_deg, dose=dose,
                                    noise_sigma=noise_sigma))


def perovskite_110(shape=(256, 256), row_spacing=24.4, pair_spacing=25.8,
                   displacement=(0.0, -0.8), reference_fraction=0.5,
                   amplitude_ref=1.0, amplitude_target=0.45, sigma_ref=2.6,
                   sigma_target=2.2, background=0.05, origin=None, dose=None,
                   noise_sigma=0.0, sampling=0.016, seed=None):
    """Synthetic image for the two-column (pair) reference model.

    Bright reference columns sit on a rectangular grid (``row_spacing`` along
    ``x``, ``pair_spacing`` along ``y``); a weaker target column sits a fraction
    ``reference_fraction`` of the way from each reference column to the one
    below it, plus ``displacement``. With the default ``reference_fraction =
    0.5`` this mimics the projected perovskite [110] cell used with
    :func:`polarmap.displacement.measure_pair_displacement`.

    Returns
    -------
    SyntheticImage
        ``v1`` is the row vector and ``v2`` the pair vector.
    """
    v1 = np.array([row_spacing, 0.0])
    v2 = np.array([0.0, pair_spacing])
    origin = (np.array([0.55 * row_spacing, 0.5 * pair_spacing])
              if origin is None else np.asarray(origin, float))
    pad = 3 * max(row_spacing, pair_spacing)
    ref, _ = _grid(shape, v1, v2, origin, pad)
    ideal = ref + reference_fraction * v2
    target = ideal + _displacement_values(displacement, ideal)
    image = (render_columns(ref, amplitude_ref, sigma_ref, shape, background)
             + render_columns(target, amplitude_target, sigma_target, shape, 0.0))
    if dose is not None or noise_sigma:
        image = add_noise(image, dose=dose, gaussian_sigma=noise_sigma, seed=seed)
    in_r, in_t = _inside(ref, shape), _inside(target, shape)
    return SyntheticImage(image, sampling, ref[in_r], target[in_t], ideal[in_t],
                          v1, v2, info=dict(kind="perovskite_110",
                                            reference_fraction=reference_fraction))


# ----------------------------------------------------------------------------
# Point lattices for strain analysis
# ----------------------------------------------------------------------------
@dataclass
class SyntheticLattice:
    """Column positions with a known deformation.

    Attributes
    ----------
    positions : numpy.ndarray
        ``(N, 2)`` column positions (px).
    v1, v2 : numpy.ndarray
        Lattice vectors of the *reference* (undeformed) lattice.
    true_F : numpy.ndarray
        ``(N, 2, 2)`` exact local deformation gradient of each column with
        respect to ``(v1, v2)``, defined from its nearest bonds.
    labels : numpy.ndarray
        ``(N,)`` integer region label (e.g. layer index).
    shape : tuple
        Image shape the positions belong to.
    image : numpy.ndarray or None
        Rendered image if requested.
    sampling : float
        Nominal pixel size (nm/px).
    """

    positions: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    true_F: np.ndarray
    labels: np.ndarray
    shape: tuple
    image: np.ndarray = None
    sampling: float = 0.02

    @property
    def true_strain(self):
        """``(exx, eyy, exy, omega)`` computed from :attr:`true_F`."""
        from .strain import strain_from_deformation_gradient
        return strain_from_deformation_gradient(self.true_F)


def lattice_points(shape, v1, v2, origin=(0.0, 0.0), margin=0.0):
    """Points of a perfect lattice lying inside an image.

    Returns the positions ``origin + i v1 + j v2`` with
    ``margin <= x <= width - 1 - margin`` (same for ``y``).
    """
    pts, _ = _grid(shape, v1, v2, np.asarray(origin, dtype=float), 0.0)
    h, w = shape
    keep = ((pts[:, 0] >= margin) & (pts[:, 0] <= w - 1 - margin)
            & (pts[:, 1] >= margin) & (pts[:, 1] <= h - 1 - margin))
    return pts[keep]


def _finish(positions, v1, v2, true_F, labels, shape, render, sigma, amplitude,
            dose, noise_sigma, seed, jitter, sampling):
    rng = np.random.default_rng(seed)
    if jitter:
        positions = positions + rng.normal(0.0, jitter, positions.shape)
    image = None
    if render:
        image = render_columns(positions, amplitude, sigma, shape, 0.05)
        if dose is not None or noise_sigma:
            image = add_noise(image, dose=dose, gaussian_sigma=noise_sigma, seed=rng)
    return SyntheticLattice(positions, np.asarray(v1, float), np.asarray(v2, float),
                            true_F, labels, tuple(shape), image, sampling)


def strained_lattice(shape=(300, 300), a=16.0, c=None, F=((1.0, 0.0), (0.0, 1.0)),
                     rotation_deg=0.0, margin=4.0, jitter=0.0, render=False,
                     sigma=2.2, amplitude=1.0, dose=None, noise_sigma=0.0,
                     sampling=0.02, seed=None):
    """A homogeneously deformed square/rectangular lattice.

    The reference lattice has vectors ``v1 = a (cos t, sin t)`` and
    ``v2 = c (-sin t, cos t)``; every position is mapped by the constant
    deformation gradient ``F`` about the image centre. ``jitter`` adds random
    position noise (px) to mimic localisation errors.

    Returns
    -------
    SyntheticLattice
    """
    c = a if c is None else c
    t = np.deg2rad(rotation_deg)
    v1 = a * np.array([np.cos(t), np.sin(t)])
    v2 = c * np.array([-np.sin(t), np.cos(t)])
    F = np.asarray(F, dtype=float)
    centre = 0.5 * np.array([shape[1] - 1, shape[0] - 1])
    # Reference points on a lattice twice the image size (centred on the
    # image centre) are deformed and the ones landing inside the image kept.
    big = (int(shape[0] * 2), int(shape[1] * 2))
    ref = lattice_points(big, v1, v2, origin=(0.3 * a, 0.3 * c))
    ref = ref - 0.5 * np.array([big[1] - 1, big[0] - 1])
    pts = centre + ref @ F.T
    h, w = shape
    keep = ((pts[:, 0] >= margin) & (pts[:, 0] <= w - 1 - margin)
            & (pts[:, 1] >= margin) & (pts[:, 1] <= h - 1 - margin))
    pts = pts[keep]
    true_F = np.broadcast_to(F, (len(pts), 2, 2)).copy()
    labels = np.zeros(len(pts), dtype=int)
    return _finish(pts, v1, v2, true_F, labels, shape, render, sigma, amplitude,
                   dose, noise_sigma, seed, jitter, sampling)


def superlattice(shape=(400, 300), a=16.0, c_values=(16.0, 16.8),
                 cells_per_layer=6, margin=4.0, jitter=0.0, render=False,
                 sigma=2.2, amplitude=1.0, dose=None, noise_sigma=0.0,
                 sampling=0.02, seed=None):
    """A coherent multilayer: common in-plane spacing, layer-dependent ``c``.

    Layers stack along ``y`` (image rows). Layer ``k`` has out-of-plane
    spacing ``c_values[k % len(c_values)]`` and ``cells_per_layer`` unit cells;
    the in-plane spacing ``a`` is the same everywhere (coherent interfaces).

    The ground truth is expressed relative to the reference lattice
    ``v1 = (a, 0)``, ``v2 = (0, c_values[0])``: for each column,
    ``F_yy = (c_up + c_down) / (2 c_values[0])`` where ``c_up`` and ``c_down``
    are the spacings to its neighbours above and below (the symmetric
    difference used by the bond-based estimators), and ``F_xx = 1``.

    Returns
    -------
    SyntheticLattice
        ``labels`` holds the layer index of every column.
    """
    h, w = shape
    c_values = np.asarray(c_values, dtype=float)
    ys, labels = [], []
    y, k = margin, 0
    while y <= h - 1 - margin:
        c = c_values[k % len(c_values)]
        for _ in range(cells_per_layer):
            if y > h - 1 - margin:
                break
            ys.append(y)
            labels.append(k)
            y += c
        k += 1
    ys, labels = np.array(ys), np.array(labels)
    gaps = np.diff(ys)
    up = np.concatenate([[gaps[0]], gaps])      # spacing to the row above
    down = np.concatenate([gaps, [gaps[-1]]])   # spacing to the row below
    fyy_row = 0.5 * (up + down) / c_values[0]
    xs = np.arange(margin, w - margin, a)

    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    row = np.repeat(np.arange(len(ys)), len(xs))
    true_F = np.zeros((len(pts), 2, 2))
    true_F[:, 0, 0] = 1.0
    true_F[:, 1, 1] = fyy_row[row]
    v1 = np.array([a, 0.0])
    v2 = np.array([0.0, c_values[0]])
    return _finish(pts, v1, v2, true_F, labels[row], shape, render, sigma,
                   amplitude, dose, noise_sigma, seed, jitter, sampling)
