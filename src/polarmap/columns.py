"""Atomic-column detection and sub-pixel position refinement.

The functions in this module are independent of the crystal structure: they
locate intensity maxima and refine their positions. Everything that depends on
the material (which columns form the reference, where the polar column should
sit) lives in :mod:`polarmap.displacement`.

Coordinate convention
---------------------
Positions are ``(x, y)`` pairs in pixels: ``x`` is the column index (increasing
to the right) and ``y`` the row index (increasing **downwards**), i.e. the
usual image convention. Arrays of positions have shape ``(N, 2)``.

Contrast modes
--------------
All routines expect atomic columns to be *bright*. :func:`as_bright_atoms`
converts an image accordingly: annular dark-field (ADF/HAADF) and integrated
DPC (iDPC) images are used as they are, while annular bright-field (ABF) and
bright-field (BF) images, in which columns are dark, are inverted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial import KDTree

__all__ = [
    "BRIGHT_MODES",
    "DARK_MODES",
    "as_bright_atoms",
    "flatten_background",
    "detect_peaks",
    "refine_gaussian",
    "refine_com",
    "compare_refinements",
    "remove_columns_gaussian",
    "IntensitySplit",
    "split_by_amplitude",
    "two_means_threshold",
]

#: Imaging modes in which atomic columns appear bright (used unchanged).
BRIGHT_MODES = frozenset({"ADF", "HAADF", "LAADF", "MAADF", "DF", "IDPC",
                          "DPC", "BF-CORRECTED"})
#: Imaging modes in which atomic columns appear dark (inverted before use).
DARK_MODES = frozenset({"ABF", "BF"})


# ----------------------------------------------------------------------------
# Contrast handling
# ----------------------------------------------------------------------------
def as_bright_atoms(image, image_mode="ADF"):
    """Return a float copy of ``image`` in which atomic columns are bright.

    Parameters
    ----------
    image : array_like
        2-D image.
    image_mode : str, default "ADF"
        Imaging mode (case-insensitive). One of :data:`BRIGHT_MODES`
        (``"ADF"``, ``"HAADF"``, ``"iDPC"``, ...), used as is, or
        :data:`DARK_MODES` (``"ABF"``, ``"BF"``), which are inverted as
        ``max(image) - image``.

    Returns
    -------
    numpy.ndarray
        ``float64`` image with bright columns.

    Notes
    -----
    This single switch lets the same detection and refinement code run on
    HAADF, iDPC (where light and heavy columns have comparable intensity) and
    ABF data. It does not correct the physics of the contrast (e.g. the
    non-linear thickness dependence of ABF intensities); it only fixes the sign.
    """
    img = np.array(image, dtype=float)          # always a copy
    if img.ndim != 2:
        raise ValueError(f"image must be 2-D, got shape {img.shape}")
    mode = str(image_mode).strip().upper()
    if mode in DARK_MODES:
        return img.max() - img
    if mode in BRIGHT_MODES:
        return img
    raise ValueError(
        f"unknown image_mode {image_mode!r}; use one of "
        f"{sorted(BRIGHT_MODES | DARK_MODES)}")


def flatten_background(image, sigma, image_mode="ADF"):
    """High-pass filter an image and rescale it to ``[0, 1]``.

    A Gaussian-blurred copy (standard deviation ``sigma`` pixels) is subtracted
    to remove slowly varying intensity: illumination, thickness gradients or the
    low-frequency background of iDPC images. This is what makes a single global
    threshold meaningful on real images.

    Parameters
    ----------
    image : array_like
        2-D image.
    sigma : float
        Width of the background filter in pixels. It should span a couple of
        lattice spacings so that the atomic columns survive the filtering.
    image_mode : str, default "ADF"
        See :func:`as_bright_atoms`.

    Returns
    -------
    numpy.ndarray
        Filtered image scaled to ``[0, 1]``.
    """
    img = as_bright_atoms(image, image_mode)
    hp = img - ndimage.gaussian_filter(img, sigma)
    hp -= hp.min()
    top = hp.max()
    return hp / top if top > 0 else hp


# ----------------------------------------------------------------------------
# Peak detection
# ----------------------------------------------------------------------------
def detect_peaks(image, min_distance, image_mode="ADF", threshold_rel=0.15,
                 flatten=True, bg_sigma=None, exclude_border=0):
    """Detect atomic columns as local intensity maxima.

    A pixel is a peak if it is the maximum of the ``min_distance`` x
    ``min_distance`` window centred on it and exceeds ``threshold_rel`` of the
    intensity range. Works on any periodic lattice.

    Parameters
    ----------
    image : array_like
        2-D image.
    min_distance : float
        Minimum spacing between columns, in pixels. Set it close to the spacing
        of the sublattice you want (e.g. the A-A spacing to pick only the A
        columns of a perovskite).
    image_mode : str, default "ADF"
        See :func:`as_bright_atoms`.
    threshold_rel : float, default 0.15
        Peaks below ``min + threshold_rel * (max - min)`` of the (flattened)
        image are discarded. Lower it if columns are missed.
    flatten : bool, default True
        Subtract a smooth background first (:func:`flatten_background`).
    bg_sigma : float, optional
        Background filter width; defaults to ``2 * min_distance``.
    exclude_border : float, default 0
        Discard peaks closer than this many pixels to the image border.

    Returns
    -------
    numpy.ndarray
        ``(N, 2)`` array of ``(x, y)`` seed positions in pixels (integer-valued
        except where tied maxima were merged).

    Notes
    -----
    Exactly equal neighbouring maxima (common in 8-bit or saturated images)
    would all pass the local-maximum test; such ties closer than half the window
    are merged into their centroid so that each column yields one seed.
    """
    img = as_bright_atoms(image, image_mode)
    if min_distance <= 0:
        raise ValueError("min_distance must be positive")
    if flatten:
        sigma = bg_sigma if bg_sigma else 2.0 * min_distance
        work = flatten_background(img, sigma, image_mode="ADF")
    else:
        work = img
    size = int(max(3, round(min_distance)))
    local_max = ndimage.maximum_filter(work, size=size, mode="nearest")
    threshold = work.min() + threshold_rel * (work.max() - work.min())
    ys, xs = np.nonzero((work == local_max) & (work >= threshold))
    peaks = _merge_tied_peaks(np.column_stack([xs, ys]).astype(float),
                              radius=0.5 * size)
    if exclude_border > 0 and len(peaks):
        h, w = work.shape
        inside = ((peaks[:, 0] > exclude_border)
                  & (peaks[:, 0] < w - exclude_border)
                  & (peaks[:, 1] > exclude_border)
                  & (peaks[:, 1] < h - exclude_border))
        peaks = peaks[inside]
    return peaks


def _merge_tied_peaks(peaks, radius):
    """Replace groups of peaks closer than ``radius`` by their centroid."""
    if len(peaks) < 2:
        return peaks
    pairs = KDTree(peaks).query_pairs(r=radius, output_type="ndarray")
    if len(pairs) == 0:
        return peaks
    parent = np.arange(len(peaks))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(len(peaks))])
    _, labels = np.unique(roots, return_inverse=True)
    merged = np.zeros((labels.max() + 1, 2))
    np.add.at(merged, labels, peaks)
    return merged / np.bincount(labels)[:, None]


# ----------------------------------------------------------------------------
# Sub-pixel refinement
# ----------------------------------------------------------------------------
def _gauss2d(p, x, y):
    """Elliptical 2-D Gaussian plus constant background.

    ``p = [amplitude, x0, y0, sigma_x, sigma_y, theta, background]``.
    """
    amp, x0, y0, sx, sy, theta, bg = p
    ct, st = np.cos(theta), np.sin(theta)
    xr = (x - x0) * ct + (y - y0) * st
    yr = -(x - x0) * st + (y - y0) * ct
    return amp * np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2)) + bg


def _gauss2d_jac(p, x, y):
    """Analytic Jacobian of :func:`_gauss2d` with respect to ``p`` (flattened)."""
    amp, x0, y0, sx, sy, theta, _ = p
    ct, st = np.cos(theta), np.sin(theta)
    dx, dy = (x - x0).ravel(), (y - y0).ravel()
    xr = dx * ct + dy * st
    yr = -dx * st + dy * ct
    e = np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
    ae = amp * e
    return np.column_stack([
        e,                                                   # d/d amplitude
        ae * (xr * ct / sx ** 2 - yr * st / sy ** 2),        # d/d x0
        ae * (xr * st / sx ** 2 + yr * ct / sy ** 2),        # d/d y0
        ae * xr ** 2 / sx ** 3,                              # d/d sigma_x
        ae * yr ** 2 / sy ** 3,                              # d/d sigma_y
        ae * xr * yr * (1.0 / sy ** 2 - 1.0 / sx ** 2),      # d/d theta
        np.ones_like(e),                                     # d/d background
    ])


def _fit_gaussian(img, x, y, box, sigma0):
    """Fit one column in a ``(2 box + 1)^2`` window around ``(x, y)``.

    Returns the parameter vector and the window grids, or ``None`` if the
    window leaves the image, the patch has no peak, or the optimiser fails.
    The analytic Jacobian (:func:`_gauss2d_jac`) makes each fit several times
    faster than finite differences.
    """
    h, w = img.shape
    xi, yi = int(round(x)), int(round(y))
    x0, x1, y0, y1 = xi - box, xi + box + 1, yi - box, yi + box + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    patch = img[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    bg_guess = np.percentile(patch, 10)
    amp_guess = patch.max() - bg_guess
    if not amp_guess > 0:
        return None
    p0 = [amp_guess, x, y, sigma0, sigma0, 0.0, bg_guess]
    lower = [0.0, x - box, y - box, 0.5, 0.5, -np.pi,
             patch.min() - abs(amp_guess)]
    upper = [3 * amp_guess + 1e-9, x + box, y + box, 3 * box, 3 * box,
             np.pi, patch.max() + 1e-9]
    try:
        res = least_squares(lambda p: (_gauss2d(p, xx, yy) - patch).ravel(),
                            p0, jac=lambda p: _gauss2d_jac(p, xx, yy),
                            bounds=(lower, upper), method="trf", max_nfev=200)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return None
    return res.x, xx, yy


def refine_gaussian(image, positions, box=6, sigma0=2.0, image_mode="ADF",
                    return_params=False):
    """Refine column positions by fitting an elliptical 2-D Gaussian.

    Each column is fitted in a ``(2 box + 1) x (2 box + 1)`` window by bounded
    non-linear least squares (``scipy.optimize.least_squares``, trust-region
    reflective) with the model

    .. math::

        I(x, y) = A \\exp\\left[-\\tfrac{1}{2}\\left(\\frac{x_r^2}{\\sigma_x^2}
                  + \\frac{y_r^2}{\\sigma_y^2}\\right)\\right] + B,

    where :math:`(x_r, y_r)` are the coordinates relative to the centre
    rotated by the angle :math:`\\theta`. The centre is constrained to stay
    within the window.

    Parameters
    ----------
    image : array_like
        2-D image.
    positions : array_like
        ``(N, 2)`` seed positions ``(x, y)`` in pixels.
    box : int, default 6
        Half-width of the fitting window in pixels. It should cover the column
        but not reach the neighbouring columns (about 1/3 of the spacing).
    sigma0 : float, default 2.0
        Initial Gaussian width in pixels.
    image_mode : str, default "ADF"
        See :func:`as_bright_atoms`.
    return_params : bool, default False
        Also return the per-column fit parameters.

    Returns
    -------
    refined : numpy.ndarray
        ``(N, 2)`` refined positions. Where a fit is not possible (window
        outside the image, failed optimisation, or centre leaving the window)
        the seed is returned unchanged and ``params["success"]`` is ``False``.
    params : dict of numpy.ndarray, optional
        Only if ``return_params``: ``amplitude``, ``sigma_x``, ``sigma_y``,
        ``ellipticity`` (= max(sigma)/min(sigma)) and ``success``. Parameters
        of unsuccessful fits are NaN.
    """
    img = as_bright_atoms(image, image_mode)
    pos = _as_positions(positions)
    refined = pos.copy()
    n = len(pos)
    amp, sxa, sya = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    success = np.zeros(n, dtype=bool)

    for i, (x, y) in enumerate(pos):
        fit = _fit_gaussian(img, x, y, box, sigma0)
        if fit is None:
            continue
        p = fit[0]
        if abs(p[1] - x) <= box and abs(p[2] - y) <= box:
            refined[i] = p[1], p[2]
            amp[i], sxa[i], sya[i] = p[0], p[3], p[4]
            success[i] = True

    if not return_params:
        return refined
    params = dict(amplitude=amp, sigma_x=sxa, sigma_y=sya,
                  ellipticity=np.fmax(sxa, sya) / np.fmin(sxa, sya),
                  success=success)
    return refined, params


def refine_com(image, positions, box=4, image_mode="ADF"):
    """Refine column positions by the intensity centre of mass.

    The centroid is computed in a ``(2 box + 1)^2`` window (clipped at the image
    border) after subtracting the window minimum. Faster than the Gaussian fit
    and a useful independent cross-check, but less robust when neighbouring
    columns enter the window.

    Parameters
    ----------
    image, positions, image_mode
        As in :func:`refine_gaussian`.
    box : int, default 4
        Half-width of the window in pixels.

    Returns
    -------
    numpy.ndarray
        ``(N, 2)`` refined positions; seeds are kept where the window is empty.
    """
    img = as_bright_atoms(image, image_mode)
    h, w = img.shape
    pos = _as_positions(positions)
    refined = pos.copy()
    for i, (x, y) in enumerate(pos):
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(xi - box, 0), min(xi + box + 1, w)
        y0, y1 = max(yi - box, 0), min(yi + box + 1, h)
        if x0 >= x1 or y0 >= y1:
            continue
        patch = img[y0:y1, x0:x1]
        patch = patch - patch.min()
        total = patch.sum()
        if total <= 0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        refined[i] = np.sum(xx * patch) / total, np.sum(yy * patch) / total
    return refined


def compare_refinements(image, positions, sampling, box=6, image_mode="ADF"):
    """Agreement between Gaussian-fit and centre-of-mass positions.

    The two estimators use different assumptions about the column shape, so
    their disagreement is a practical proxy for the localisation precision. It
    is not an accuracy measure: residual aberrations, mistilt and channelling
    can shift the *apparent* column position of both estimators alike.

    Parameters
    ----------
    image, positions, image_mode
        As in :func:`refine_gaussian`.
    sampling : float
        Pixel size in nm/px (used to express the result in pm).
    box : int, default 6
        Gaussian window half-width; the centre-of-mass window uses
        ``max(box - 2, 2)``.

    Returns
    -------
    dict
        ``rms_pm`` and ``median_pm`` (disagreement over successfully fitted
        columns), ``n_compared``, and the two position arrays ``gaussian`` and
        ``com``.
    """
    g, params = refine_gaussian(image, positions, box=box,
                                image_mode=image_mode, return_params=True)
    c = refine_com(image, positions, box=max(box - 2, 2), image_mode=image_mode)
    ok = params["success"]
    d = np.hypot(g[ok, 0] - c[ok, 0], g[ok, 1] - c[ok, 1]) * sampling * 1e3
    return dict(rms_pm=float(np.sqrt(np.mean(d ** 2))) if d.size else np.nan,
                median_pm=float(np.median(d)) if d.size else np.nan,
                n_compared=int(ok.sum()), gaussian=g, com=c)


def remove_columns_gaussian(image, positions, box=6, sigma0=2.0,
                            image_mode="ADF"):
    """Subtract a fitted 2-D Gaussian from every listed column.

    The fitted background constant is kept, so the result is a residual image
    in which the listed columns have been removed. This is the key step for
    measuring a *weak* column next to strong ones: on real (blurred) images the
    tails of the bright reference columns bleed into the cell centre and would
    otherwise pull the fit of the weak central column.

    Parameters
    ----------
    image, positions, box, sigma0, image_mode
        As in :func:`refine_gaussian`. Columns whose window leaves the image or
        whose fit fails are left untouched.

    Returns
    -------
    numpy.ndarray
        Residual image (bright-atom convention).
    """
    img = as_bright_atoms(image, image_mode)
    out = img.copy()
    for x, y in _as_positions(positions):
        fit = _fit_gaussian(img, x, y, box, sigma0)
        if fit is None:
            continue
        p, xx, yy = fit
        xi, yi = int(round(x)), int(round(y))
        out[yi - box:yi + box + 1, xi - box:xi + box + 1] -= (
            _gauss2d(p, xx, yy) - p[6])
    return out


# ----------------------------------------------------------------------------
# Separating two sublattices by intensity
# ----------------------------------------------------------------------------
@dataclass
class IntensitySplit:
    """Result of :func:`split_by_amplitude`.

    Attributes
    ----------
    bright, dim : numpy.ndarray
        ``(N, 2)`` refined positions of the bright and dim columns.
    bright_amplitude, dim_amplitude : numpy.ndarray
        Fitted Gaussian amplitudes of those columns.
    threshold : float
        Amplitude threshold separating the two groups.
    """

    bright: np.ndarray
    dim: np.ndarray
    bright_amplitude: np.ndarray
    dim_amplitude: np.ndarray
    threshold: float


def two_means_threshold(values, tol=1e-8, max_iter=100):
    """Threshold splitting 1-D data into two groups (iterative 2-means).

    Starting from the median, the threshold is moved to the midpoint of the two
    group means until it no longer changes.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("no finite values to split")
    t = float(np.median(values))
    for _ in range(max_iter):
        low, high = values[values < t], values[values >= t]
        if low.size == 0 or high.size == 0:
            break
        new_t = 0.5 * (low.mean() + high.mean())
        if abs(new_t - t) < tol:
            t = new_t
            break
        t = new_t
    return float(t)


def split_by_amplitude(image, seeds, box=3, image_mode="ADF", edge=0):
    """Refine all detected columns and split them into bright and dim groups.

    Useful when two sublattices differ in intensity but not by a wide margin,
    e.g. the Pb and Ti/O columns of a perovskite [110] projection. Each seed is
    refined by :func:`refine_gaussian`; failed fits and columns within ``edge``
    pixels of the border are discarded; the fitted amplitudes are then split
    with :func:`two_means_threshold`.

    Parameters
    ----------
    image : array_like
        2-D image.
    seeds : array_like
        ``(N, 2)`` seed positions of *all* columns.
    box : int, default 3
        Gaussian fitting half-width.
    image_mode : str, default "ADF"
        See :func:`as_bright_atoms`.
    edge : float, default 0
        Border exclusion in pixels.

    Returns
    -------
    IntensitySplit
        Refined (sub-pixel) positions and amplitudes of both groups.
    """
    refined, params = refine_gaussian(image, seeds, box=box,
                                      image_mode=image_mode, return_params=True)
    amp = params["amplitude"]
    h, w = np.asarray(image).shape
    valid = (params["success"] & np.isfinite(amp) & (amp > 0)
             & (refined[:, 0] > edge) & (refined[:, 0] < w - edge)
             & (refined[:, 1] > edge) & (refined[:, 1] < h - edge))
    refined, amp = refined[valid], amp[valid]
    if len(refined) < 4:
        raise RuntimeError(
            "Too few valid columns for an intensity split; lower the detection "
            "threshold or check image_mode.")
    threshold = two_means_threshold(amp)
    bright = amp >= threshold
    return IntensitySplit(refined[bright], refined[~bright],
                          amp[bright], amp[~bright], threshold)


def _as_positions(positions):
    pos = np.array(positions, dtype=float)
    if pos.size == 0:
        return pos.reshape(0, 2)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError(f"positions must have shape (N, 2), got {pos.shape}")
    return pos
