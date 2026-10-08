"""Angle conventions, circular statistics and robust outlier rules.

Angle conventions
-----------------
Displacements are stored in image axes: ``u`` along ``+x`` (right) and ``v``
along ``+y`` (**down**). Two angle conventions are supported everywhere:

``"cartesian"`` (default)
    ``theta = atan2(-v, u)``: 0 deg points right, +90 deg points **up**,
    angles increase counter-clockwise as seen on the screen. This is the
    convention used for statistics and colour coding in the manuscript.
``"image"``
    ``theta = atan2(v, u)``: 0 deg points right, +90 deg points **down**,
    angles increase clockwise as seen on the screen.

Angles are returned in ``(-180, 180]`` degrees.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ANGLE_CONVENTIONS",
    "displacement_angle",
    "angular_difference",
    "circular_statistics",
    "mad_outlier_mask",
    "local_outlier_mask",
    "describe_displacements",
    "format_descriptors",
]

ANGLE_CONVENTIONS = ("cartesian", "image")


def _check_convention(convention):
    if convention not in ANGLE_CONVENTIONS:
        raise ValueError(f"convention must be one of {ANGLE_CONVENTIONS}")


def displacement_angle(u, v, convention="cartesian", degrees=True):
    """Orientation of displacement vectors.

    Parameters
    ----------
    u, v : array_like
        Displacement components in image axes (``v`` positive downwards).
    convention : {"cartesian", "image"}, default "cartesian"
        See the module documentation.
    degrees : bool, default True
        Return degrees (otherwise radians).

    Returns
    -------
    numpy.ndarray
        Angles in ``(-180, 180]`` degrees (or ``(-pi, pi]`` radians).
    """
    _check_convention(convention)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    ang = np.arctan2(-v if convention == "cartesian" else v, u)
    # atan2(-0.0, x < 0) gives -pi; map it to +pi so the range is (-pi, pi]
    ang = np.where(ang <= -np.pi, ang + 2 * np.pi, ang)
    return np.degrees(ang) if degrees else ang


def angular_difference(a, b, degrees=True):
    """Smallest signed difference ``a - b`` between two angles.

    Returns values in ``[-180, 180)`` degrees (or ``[-pi, pi)`` radians), so
    that e.g. 179 deg and -179 deg differ by -2 deg, not 358 deg.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    half = 180.0 if degrees else np.pi
    return np.mod(a - b + half, 2 * half) - half


def circular_statistics(angles, degrees=True):
    """Mean direction and spread of a set of angles.

    With unit vectors ``(cos t_i, sin t_i)``, the mean resultant vector has
    length ``R`` (0 = uniformly spread, 1 = all identical) and direction equal
    to the circular mean. The circular standard deviation is
    ``sqrt(-2 ln R)``.

    Parameters
    ----------
    angles : array_like
        Angles (NaNs are ignored).
    degrees : bool, default True
        Input and output in degrees.

    Returns
    -------
    dict
        ``mean`` (circular mean), ``std`` (circular standard deviation),
        ``resultant_length`` (``R``) and ``n``.
    """
    ang = np.asarray(angles, dtype=float).ravel()
    ang = ang[np.isfinite(ang)]
    if ang.size == 0:
        return dict(mean=np.nan, std=np.nan, resultant_length=np.nan, n=0)
    rad = np.deg2rad(ang) if degrees else ang
    c, s = np.mean(np.cos(rad)), np.mean(np.sin(rad))
    r = float(np.hypot(c, s))
    mean = float(np.arctan2(s, c))
    std = float(np.sqrt(-2.0 * np.log(max(r, 1e-12))))
    if degrees:
        mean, std = float(np.degrees(mean)), float(np.degrees(std))
    return dict(mean=mean, std=std, resultant_length=r, n=int(ang.size))


def mad_outlier_mask(values, n_mad, *, two_sided=True, normal_consistent=True):
    """Flag outliers with a median / median-absolute-deviation (MAD) rule.

    A value ``x`` is an outlier if ``|x - median| > n_mad * s`` (two-sided) or
    ``x - median > n_mad * s`` (upper side only), where ``s`` is the MAD,
    multiplied by 1.4826 when ``normal_consistent`` so that it estimates the
    standard deviation of normally distributed data. Unlike a percentile clip,
    this removes nothing when no value is extreme.

    Parameters
    ----------
    values : array_like
        Data; NaNs are ignored and never flagged.
    n_mad : float
        Threshold in units of ``s``.
    two_sided : bool, default True
        Flag both tails (otherwise only values above the median).
    normal_consistent : bool, default True
        Scale the MAD by 1.4826.

    Returns
    -------
    numpy.ndarray
        Boolean mask, ``True`` for outliers.
    """
    x = np.asarray(values, dtype=float)
    finite = np.isfinite(x)
    out = np.zeros(x.shape, dtype=bool)
    if not finite.any():
        return out
    med = np.median(x[finite])
    s = np.median(np.abs(x[finite] - med))
    if normal_consistent:
        s *= 1.4826
    s += 1e-12
    dev = x[finite] - med
    out[finite] = (np.abs(dev) if two_sided else dev) > n_mad * s
    return out


def local_outlier_mask(positions, values, n_mad=5.0, k=8):
    """Flag values that deviate from their spatial neighbourhood.

    Each value is compared with the median of its ``k`` nearest neighbours; the
    residuals are then tested with :func:`mad_outlier_mask` (two-sided,
    normal-consistent). Spatially coherent structure, such as a strained layer,
    a domain or a gradient, is removed by the local median and is therefore
    *not* flagged; only isolated deviations are. Prefer this to a global rule
    for any map that is expected to vary across the field of view: a global
    median/MAD test flags a strained layer as a whole when it covers less than
    about half of the columns.

    Parameters
    ----------
    positions : array_like
        ``(N, 2)`` positions.
    values : array_like
        ``(N,)`` values; NaNs are ignored and never flagged.
    n_mad : float, default 5
        Threshold in robust standard deviations of the residuals.
    k : int, default 8
        Number of neighbours (8 = first and second shell of a square lattice).

    Returns
    -------
    numpy.ndarray
        Boolean mask, ``True`` for outliers.
    """
    from scipy.spatial import KDTree

    pos = np.asarray(positions, dtype=float).reshape(-1, 2)
    vals = np.asarray(values, dtype=float)
    ok = np.isfinite(vals) & np.all(np.isfinite(pos), axis=1)
    out = np.zeros(vals.shape, dtype=bool)
    if ok.sum() <= k:
        out[ok] = mad_outlier_mask(vals[ok], n_mad)
        return out
    v = vals[ok]
    _, idx = KDTree(pos[ok]).query(pos[ok], k=k + 1)
    residual = v - np.median(v[idx[:, 1:]], axis=1)
    out[ok] = mad_outlier_mask(residual, n_mad, two_sided=True,
                               normal_consistent=True)
    return out


def describe_displacements(field, sampling, n_mad=6.0, convention="cartesian"):
    """Statistical descriptors of a displacement field.

    Magnitudes above ``median + n_mad * MAD`` (raw MAD, upper side only) are
    excluded first; this is the rule used for the descriptors and histograms in
    the manuscript.

    Parameters
    ----------
    field : polarmap.displacement.DisplacementField
        Measured displacements.
    sampling : float
        Pixel size in nm/px.
    n_mad : float or None, default 6
        Outlier threshold; ``None`` keeps every finite vector.
    convention : {"cartesian", "image"}, default "cartesian"
        Angle convention (see the module documentation).

    Returns
    -------
    dict
        ``n``, ``n_excluded``, magnitude statistics in pm (``mean_pm``,
        ``median_pm``, ``std_pm``, ``mad_pm``, ``min_pm``, ``max_pm``),
        ``circular_mean_deg``, ``circular_std_deg``, ``resultant_length``,
        and ``kept`` (boolean mask of the vectors used).
    """
    mag = np.asarray(field.magnitude, dtype=float) * sampling * 1e3
    ang = field.angle(convention=convention)
    kept = np.isfinite(mag)
    if n_mad is not None:
        kept &= ~mad_outlier_mask(mag, n_mad, two_sided=False,
                                  normal_consistent=False)
    m = mag[kept]
    med = float(np.median(m)) if m.size else np.nan
    circ = circular_statistics(ang[kept])
    return dict(
        n=int(kept.sum()),
        n_excluded=int((~kept).sum()),
        mean_pm=float(m.mean()) if m.size else np.nan,
        median_pm=med,
        std_pm=float(m.std()) if m.size else np.nan,
        mad_pm=float(np.median(np.abs(m - med))) if m.size else np.nan,
        min_pm=float(m.min()) if m.size else np.nan,
        max_pm=float(m.max()) if m.size else np.nan,
        circular_mean_deg=circ["mean"],
        circular_std_deg=circ["std"],
        resultant_length=circ["resultant_length"],
        convention=convention,
        kept=kept,
    )


def format_descriptors(desc):
    """Format the output of :func:`describe_displacements` as text."""
    return (
        f"Displacement descriptors (n = {desc['n']}, "
        f"{desc['n_excluded']} excluded as outliers)\n"
        f"  magnitude |d| (pm): mean {desc['mean_pm']:.1f}  median "
        f"{desc['median_pm']:.1f}  std {desc['std_pm']:.1f}  MAD "
        f"{desc['mad_pm']:.1f}  min {desc['min_pm']:.1f}  max {desc['max_pm']:.1f}\n"
        f"  orientation ({desc['convention']}, deg): circular mean "
        f"{desc['circular_mean_deg']:.1f}  circular std "
        f"{desc['circular_std_deg']:.1f}  (R = {desc['resultant_length']:.2f})"
    )
