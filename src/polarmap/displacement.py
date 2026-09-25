"""Projected atomic-column displacement mapping.

A displacement map is always *the position of a target (polar) column minus the
position it would occupy in a centrosymmetric reference*. The column
localisation (:mod:`polarmap.columns`) is generic; what depends on the crystal
structure and projection is how that reference position is built from the
surrounding reference columns. This module implements the reference models:

* **perovskite [100]**: four A columns forming a complete cage; the ideal
  target position is the centroid of the four measured corners
  (:func:`measure_cage_displacement`);
* **perovskite [110]**: two A columns along the projected polar axis (a
  complete pair); the ideal position is ``(1 - f) r0 + f r1`` with ``f = 1/2``
  (:func:`measure_pair_displacement`);
* **wurtzite-like dumbbells**: one reference column; the ideal position is the
  reference plus a fixed basis offset (:func:`displacement_from_offset`).

The measured quantity is an image-derived displacement (pixels, or pm after
multiplication by the sampling). It is **not** an absolute polarization; see
:func:`displacement_to_polarization` for a semi-quantitative conversion.

All functions return a :class:`DisplacementField`. Displacements are given in
image axes (``u`` right, ``v`` down) and point *from the reference position to
the measured target column*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import KDTree

from .columns import (as_bright_atoms, refine_com, refine_gaussian,
                      remove_columns_gaussian)
from .geometry import inside_image
from .statistics import displacement_angle

__all__ = [
    "DisplacementField",
    "predict_second_sublattice",
    "find_complete_perovskite_cages",
    "measure_cage_displacement",
    "find_complete_reference_pairs",
    "local_residual_maxima",
    "measure_pair_displacement",
    "displacement_from_sublattices",
    "displacement_from_cage",
    "displacement_from_offset",
    "displacement_to_polarization",
]


# ----------------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------------
@dataclass
class DisplacementField:
    """A set of displacement vectors anchored at reference positions.

    Attributes
    ----------
    x, y : numpy.ndarray
        Reference (ideal, centrosymmetric) positions in pixels.
    u, v : numpy.ndarray
        Displacement components in pixels, image axes (``v`` positive down):
        ``(u, v) = measured target - reference``.
    per_site : dict of numpy.ndarray
        Optional per-vector data with the same length (e.g. ``amplitude``,
        ``corner_indices``). Subsetting keeps them aligned.
    diagnostics : dict
        Optional extra information that is not per vector (e.g. the residual
        image the target columns were fitted on).
    """

    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    per_site: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float).ravel()
        self.y = np.asarray(self.y, dtype=float).ravel()
        self.u = np.asarray(self.u, dtype=float).ravel()
        self.v = np.asarray(self.v, dtype=float).ravel()
        n = len(self.x)
        if not len(self.y) == len(self.u) == len(self.v) == n:
            raise ValueError("x, y, u and v must have the same length")
        for key, value in self.per_site.items():
            if len(value) != n:
                raise ValueError(f"per_site[{key!r}] has length {len(value)}, "
                                 f"expected {n}")

    def __len__(self):
        return len(self.x)

    @property
    def reference_xy(self):
        """``(N, 2)`` reference positions."""
        return np.column_stack([self.x, self.y])

    @property
    def target_xy(self):
        """``(N, 2)`` measured target positions (reference + displacement)."""
        return np.column_stack([self.x + self.u, self.y + self.v])

    @property
    def magnitude(self):
        """Displacement magnitude in pixels."""
        return np.hypot(self.u, self.v)

    def magnitude_in(self, sampling, unit="pm"):
        """Displacement magnitude in physical units.

        Parameters
        ----------
        sampling : float
            Pixel size in nm/px.
        unit : {"pm", "nm", "angstrom"}, default "pm"
        """
        factor = {"pm": 1e3, "nm": 1.0, "angstrom": 10.0}[unit]
        return self.magnitude * sampling * factor

    def angle(self, convention="cartesian", degrees=True):
        """Displacement orientation; see :func:`polarmap.statistics.displacement_angle`."""
        return displacement_angle(self.u, self.v, convention=convention,
                                  degrees=degrees)

    def subset(self, mask):
        """Return a new field with only the vectors selected by ``mask``."""
        mask = np.asarray(mask)
        return DisplacementField(
            self.x[mask], self.y[mask], self.u[mask], self.v[mask],
            per_site={k: np.asarray(val)[mask] for k, val in self.per_site.items()},
            diagnostics=dict(self.diagnostics))

    def with_sign(self, sign):
        """Return a copy with the displacement multiplied by ``sign`` (±1)."""
        if sign not in (1, -1):
            raise ValueError("sign must be +1 or -1")
        return DisplacementField(self.x, self.y, sign * self.u, sign * self.v,
                                 per_site=dict(self.per_site),
                                 diagnostics=dict(self.diagnostics))

    def to_csv(self, path, sampling=None, convention="cartesian"):
        """Write the field to a CSV file.

        Columns: reference and target positions (px), displacement components
        and magnitude (px), orientation (deg, given ``convention``) and, if
        ``sampling`` is given, the components and magnitude in pm.
        """
        cols = [self.x, self.y, self.x + self.u, self.y + self.v,
                self.u, self.v, self.magnitude, self.angle(convention)]
        names = ["x_ref_px", "y_ref_px", "x_target_px", "y_target_px",
                 "u_px", "v_px", "magnitude_px", f"angle_{convention}_deg"]
        if sampling is not None:
            k = sampling * 1e3
            cols += [self.u * k, self.v * k, self.magnitude * k]
            names += ["u_pm", "v_pm", "magnitude_pm"]
        np.savetxt(path, np.column_stack(cols), delimiter=",",
                   header=",".join(names), comments="", fmt="%.6f")

    @classmethod
    def from_csv(cls, path):
        """Read a field written by :meth:`to_csv`."""
        data = np.genfromtxt(path, delimiter=",", names=True)
        data = np.atleast_1d(data)
        return cls(data["x_ref_px"], data["y_ref_px"], data["u_px"], data["v_px"])


# ----------------------------------------------------------------------------
# Perovskite [100]: complete four-corner cages
# ----------------------------------------------------------------------------
def predict_second_sublattice(A, v1, v2, offset=(0.5, 0.5)):
    """Ideal second-sublattice positions from a global lattice.

    ``A + offset[0] * v1 + offset[1] * v2`` for every reference column. This
    extrapolates the *average* lattice and is kept for diagnostics; prefer the
    complete-cage reference, which uses the measured local cage instead.

    Parameters
    ----------
    A : array_like
        ``(N, 2)`` reference column positions.
    v1, v2 : array_like
        Lattice vectors.
    offset : (float, float), default (0.5, 0.5)
        Fractional position of the second sublattice in the cell; ``(0.5, 0.5)``
        is the body centre (B site of a perovskite [100] projection).
    """
    A = np.asarray(A, dtype=float)
    return A + offset[0] * np.asarray(v1, float) + offset[1] * np.asarray(v2, float)


def find_complete_perovskite_cages(A, v1, v2, tolerance=None):
    """Find all complete four-corner A cages and their measured centroids.

    For every A column taken as anchor ``A0``, the expected cage corners are
    ``A0``, ``A0 + v1``, ``A0 + v2`` and ``A0 + v1 + v2``. A cage is accepted
    only if a measured A column lies within ``tolerance`` of every expected
    corner, the four matched columns are distinct, and their centroid lies
    within ``tolerance`` of ``A0 + (v1 + v2) / 2``. The reference position is
    the centroid of the **four measured** corners, so edge cells, vacancies and
    missed columns are rejected automatically instead of producing a false
    reference.

    Parameters
    ----------
    A : array_like
        ``(N, 2)`` refined A-column positions (one sublattice only).
    v1, v2 : array_like
        Primitive A-A lattice vectors.
    tolerance : float, optional
        Maximum corner mismatch in pixels; default ``0.28 * min(|v1|, |v2|)``.
        Must be smaller than half the lattice spacing. Increase slightly
        (e.g. to 0.30-0.35 of the spacing) for strongly strained data.

    Returns
    -------
    centres : numpy.ndarray
        ``(M, 2)`` cage centroids.
    corner_indices : numpy.ndarray
        ``(M, 4)`` indices into ``A`` of the corners, ordered ``A0``,
        ``A0 + v1``, ``A0 + v2``, ``A0 + v1 + v2``.
    """
    A = np.asarray(A, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    if A.ndim != 2 or A.shape[1] != 2:
        raise ValueError("A must have shape (N, 2) with (x, y) coordinates")
    empty = (np.empty((0, 2)), np.empty((0, 4), dtype=int))
    if len(A) < 4:
        return empty

    a_min = min(np.linalg.norm(v1), np.linalg.norm(v2))
    if not np.isfinite(a_min) or a_min <= 0:
        raise ValueError("v1 and v2 must be finite, non-zero lattice vectors")
    tolerance = 0.28 * a_min if tolerance is None else float(tolerance)
    if tolerance <= 0 or tolerance >= 0.5 * a_min:
        raise ValueError("tolerance must be positive and smaller than half the "
                         "lattice spacing")

    tree = KDTree(A)
    offsets = np.array([[0.0, 0.0], v1, v2, v1 + v2])
    cages = {}
    for anchor in A:
        dist, idx = tree.query(anchor + offsets, k=1)
        idx = np.asarray(idx, dtype=int)
        if np.any(dist > tolerance) or np.unique(idx).size != 4:
            continue
        centre = A[idx].mean(axis=0)
        if np.linalg.norm(centre - (anchor + 0.5 * (v1 + v2))) > tolerance:
            continue
        cages.setdefault(tuple(sorted(idx.tolist())), (centre, idx))
    if not cages:
        return empty
    centres = np.vstack([c for c, _ in cages.values()])
    corners = np.vstack([i for _, i in cages.values()]).astype(int)
    return centres, corners


def measure_cage_displacement(image, A, v1, v2, *, image_mode="ADF",
                              reference="complete_cage", refine="gaussian",
                              box=None, edge=None, subtract_reference=True,
                              cage_tolerance=None, offset=(0.5, 0.5),
                              max_displacement=0.34, min_amplitude=0.30,
                              sign=1):
    """Measure the displacement of the central column of each A cage.

    Procedure (perovskite [100] projection):

    1. **Reference subtraction** -- a 2-D Gaussian is fitted to every A column
       and subtracted (:func:`~polarmap.columns.remove_columns_gaussian`,
       window half-width ``max(4, 0.45 * cell)``), so the tails of the bright A
       columns do not bias the weak central column.
    2. **Reference positions** -- centroids of complete four-corner cages
       (:func:`find_complete_perovskite_cages`).
    3. **Target refinement** -- the central (B) column is refined on the
       residual image, starting from the cage centroid, by a 2-D Gaussian fit
       (or centre of mass) in a window of half-width ``box``.
    4. **Quality control** -- a vector is kept only if the fit converged, the
       displacement is smaller than ``max_displacement * cell`` and (Gaussian
       fits) the amplitude exceeds ``min_amplitude`` times the median
       amplitude.

    Here ``cell = (|v1| + |v2|) / 2``.

    Parameters
    ----------
    image : array_like
        2-D image.
    A : array_like
        ``(N, 2)`` refined A-column positions.
    v1, v2 : array_like
        Primitive A-A lattice vectors.
    image_mode : str, default "ADF"
        See :func:`polarmap.columns.as_bright_atoms`.
    reference : {"complete_cage", "anchor"}, default "complete_cage"
        ``"anchor"`` uses :func:`predict_second_sublattice` (global lattice
        extrapolation) and is intended only for diagnostic comparison.
    refine : {"gaussian", "com"}, default "gaussian"
        Target refinement method.
    box : int, optional
        Target fitting half-width; default ``max(3, round(0.30 * cell))``.
    edge : float, optional
        Cages closer than this to the border are skipped so that the fitting
        window stays inside the image; default ``box + 1``.
    subtract_reference : bool, default True
        Perform step 1.
    cage_tolerance : float, optional
        See :func:`find_complete_perovskite_cages`.
    offset : (float, float), default (0.5, 0.5)
        Only used with ``reference="anchor"``.
    max_displacement : float, default 0.34
        Maximum accepted displacement as a fraction of ``cell``.
    min_amplitude : float, default 0.30
        Minimum fitted amplitude relative to the median (Gaussian fits only).
    sign : {1, -1}, default 1
        Multiply the displacements by this sign (``-1`` reverses the arrow
        convention, e.g. to show the direction opposite to the cation shift).

    Returns
    -------
    DisplacementField
        ``per_site`` holds ``corner_indices`` (``(M, 4)``, ``-1`` for the
        anchor reference) and, for Gaussian fits, ``amplitude``;
        ``diagnostics["residual"]`` is the image the targets were fitted on.
    """
    img = as_bright_atoms(image, image_mode)
    h, w = img.shape
    A = np.asarray(A, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    cell = 0.5 * (np.linalg.norm(v1) + np.linalg.norm(v2))
    if box is None:
        box = max(3, int(round(0.30 * cell)))
    if edge is None:
        edge = box + 1

    if subtract_reference:
        ref_box = max(4, int(round(0.45 * cell)))
        work = remove_columns_gaussian(img, A, box=ref_box, image_mode="ADF")
    else:
        work = img

    if reference == "complete_cage":
        ideal, corners = find_complete_perovskite_cages(A, v1, v2,
                                                        tolerance=cage_tolerance)
    elif reference == "anchor":
        ideal = predict_second_sublattice(A, v1, v2, offset)
        corners = np.full((len(ideal), 4), -1, dtype=int)
    else:
        raise ValueError("reference must be 'complete_cage' or 'anchor'")
    if len(ideal) == 0:
        raise RuntimeError(
            "No complete cages were found. Check that A contains only one "
            "sublattice, that v1/v2 are primitive A-A vectors, and increase "
            "cage_tolerance slightly if necessary.")

    inside = inside_image(ideal, (h, w), margin=edge)
    ideal, corners = ideal[inside], corners[inside]
    if len(ideal) == 0:
        raise RuntimeError("All cages were removed by the fitting-window edge "
                           "filter; reduce `edge` or `box`.")

    per_site = {"corner_indices": corners}
    if refine == "gaussian":
        found, params = refine_gaussian(work, ideal, box=box, image_mode="ADF",
                                        return_params=True)
        amp = params["amplitude"]
        per_site["amplitude"] = amp
    elif refine == "com":
        found = refine_com(work, ideal, box=box, image_mode="ADF")
        amp = None
    else:
        raise ValueError("refine must be 'gaussian' or 'com'")

    disp = found - ideal
    good = (np.isfinite(found).all(axis=1)
            & (np.hypot(disp[:, 0], disp[:, 1]) < max_displacement * cell))
    if amp is not None:
        amp_ok = good & np.isfinite(amp) & (amp > 0)
        if amp_ok.any():
            good &= np.isfinite(amp) & (amp > min_amplitude * np.median(amp[amp_ok]))
        else:
            good &= False

    result = DisplacementField(ideal[:, 0], ideal[:, 1], disp[:, 0], disp[:, 1],
                               per_site=per_site,
                               diagnostics={"residual": work, "cell": cell,
                                            "box": box, "reference": reference})
    return result.subset(good).with_sign(sign)


# ----------------------------------------------------------------------------
# Perovskite [110] and other two-column references: complete pairs
# ----------------------------------------------------------------------------
def find_complete_reference_pairs(reference_xy, pair_vector,
                                  reference_fraction=0.5, tolerance=None):
    """Build ideal target positions from complete two-column pairs.

    For every reference column ``r0`` the column nearest to
    ``r0 + pair_vector`` is accepted as partner ``r1`` if it lies within
    ``tolerance`` of that point (and on the positive side). The ideal target
    position is interpolated between the **two measured** endpoints,

    .. math:: \\mathbf{r}_\\mathrm{ideal} = (1 - f)\\,\\mathbf{r}_0 + f\\,\\mathbf{r}_1,

    so an edge column or a missing neighbour cannot create a false reference.

    Parameters
    ----------
    reference_xy : array_like
        ``(N, 2)`` reference column positions.
    pair_vector : array_like
        Translation from one reference column to its partner
        (e.g. from :func:`polarmap.lattice.estimate_pair_vector`).
    reference_fraction : float, default 0.5
        ``f`` above. ``0.5`` is the pair midpoint (perovskite [110]); other
        values describe an asymmetric internal coordinate, e.g. ``3/8`` for the
        cation-to-anion spacing of wurtzite along ``c``. In that case the sign
        of ``pair_vector`` decides which end is ``r0``.
    tolerance : float, optional
        Maximum partner mismatch in pixels; default ``0.28 * |pair_vector|``.

    Returns
    -------
    ideal_xy : numpy.ndarray
        ``(M, 2)`` ideal target positions.
    pair_indices : numpy.ndarray
        ``(M, 2)`` indices ``(i0, i1)`` into ``reference_xy``.
    """
    ref = np.asarray(reference_xy, dtype=float)
    pair_vector = np.asarray(pair_vector, dtype=float)
    spacing = np.linalg.norm(pair_vector)
    if spacing <= 0:
        raise ValueError("pair_vector must be non-zero")
    if not 0.0 < reference_fraction < 1.0:
        raise ValueError("reference_fraction must lie between 0 and 1")
    tolerance = 0.28 * spacing if tolerance is None else float(tolerance)
    if not 0.0 < tolerance < 0.5 * spacing:
        raise ValueError("tolerance must be positive and smaller than half the "
                         "pair spacing")
    if len(ref) < 2:
        return np.empty((0, 2)), np.empty((0, 2), dtype=int)

    dist, partner = KDTree(ref).query(ref + pair_vector, k=1)
    ideal, pairs = [], []
    for i, (d, j) in enumerate(zip(dist, partner, strict=True)):
        j = int(j)
        if j == i or d > tolerance:
            continue
        if np.dot(ref[j] - ref[i], pair_vector) <= 0:
            continue
        ideal.append((1.0 - reference_fraction) * ref[i]
                     + reference_fraction * ref[j])
        pairs.append((i, j))
    if not ideal:
        return np.empty((0, 2)), np.empty((0, 2), dtype=int)
    return np.array(ideal), np.array(pairs, dtype=int)


def local_residual_maxima(residual, ideal_xy, search_radius):
    """Brightest pixel of ``residual`` within a square window around each point.

    Used to seed the weak target column near every ideal position before the
    Gaussian refinement.

    Parameters
    ----------
    residual : array_like
        2-D image (bright atoms), usually after reference subtraction.
    ideal_xy : array_like
        ``(M, 2)`` window centres.
    search_radius : int
        Window half-width in pixels.

    Returns
    -------
    numpy.ndarray
        ``(M, 2)`` integer-valued seed positions.
    """
    residual = np.asarray(residual, dtype=float)
    h, w = residual.shape
    ideal_xy = np.asarray(ideal_xy, dtype=float).reshape(-1, 2)
    seeds = np.empty_like(ideal_xy)
    for i, (x, y) in enumerate(ideal_xy):
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - search_radius), min(w, xi + search_radius + 1)
        y0, y1 = max(0, yi - search_radius), min(h, yi + search_radius + 1)
        patch = residual[y0:y1, x0:x1]
        py, px = np.unravel_index(np.argmax(patch), patch.shape)
        seeds[i] = (x0 + px, y0 + py)
    return seeds


def measure_pair_displacement(image, reference_xy, pair_vector, *,
                              reference_fraction=0.5, image_mode="ADF",
                              pair_tolerance=None, subtract_reference=True,
                              reference_subtract_box=None,
                              weak_search_radius=None, weak_fit_box=None,
                              max_displacement=None, max_ellipticity=3.0,
                              min_amplitude=0.25, sign=1):
    """Measure weak-column displacements relative to complete reference pairs.

    Procedure (e.g. perovskite [110], where the B-containing column lies between
    two bright A columns along the projected polar axis):

    1. ideal positions from complete pairs (:func:`find_complete_reference_pairs`);
    2. the bright reference columns are removed by Gaussian subtraction;
    3. a seed for the weak column is the brightest residual pixel within
       ``weak_search_radius`` of each ideal position;
    4. the seed is refined by a 2-D Gaussian fit on the residual;
    5. a vector is kept if the fit converged, the displacement is below
       ``max_displacement``, the ellipticity below ``max_ellipticity`` and the
       amplitude above ``min_amplitude`` times the median amplitude.

    Parameters
    ----------
    image : array_like
        2-D image.
    reference_xy : array_like
        ``(N, 2)`` refined positions of the bright reference columns (for
        example ``split_by_amplitude(...).bright``).
    pair_vector : array_like
        Reference-to-partner translation.
    reference_fraction : float, default 0.5
        See :func:`find_complete_reference_pairs`.
    image_mode : str, default "ADF"
    pair_tolerance : float, optional
        Pair matching tolerance (px), default ``0.28 * |pair_vector|``.
    subtract_reference : bool, default True
    reference_subtract_box : int, optional
        Half-width for reference subtraction; default
        ``max(4, round(0.22 * |pair_vector|))``.
    weak_search_radius : int, optional
        Default ``max(2, round(0.20 * |pair_vector|))``.
    weak_fit_box : int, optional
        Default ``max(3, round(0.16 * |pair_vector|))``.
    max_displacement : float, optional
        Maximum displacement in pixels; default ``0.30 * |pair_vector|``.
    max_ellipticity : float, default 3.0
    min_amplitude : float, default 0.25
    sign : {1, -1}, default 1

    Returns
    -------
    DisplacementField
        ``per_site`` holds ``pair_indices``, ``seeds``, ``amplitude`` and
        ``ellipticity``; ``diagnostics["residual"]`` is the reference-subtracted
        image.
    """
    bright = as_bright_atoms(image, image_mode)
    ref = np.asarray(reference_xy, dtype=float)
    spacing = np.linalg.norm(pair_vector)

    ideal, pairs = find_complete_reference_pairs(
        ref, pair_vector, reference_fraction=reference_fraction,
        tolerance=pair_tolerance)
    if len(ideal) == 0:
        raise RuntimeError(
            "No complete reference pairs were found. Check the bright-column "
            "selection, the pair vector and the pair tolerance.")

    if weak_fit_box is None:
        weak_fit_box = max(3, int(round(0.16 * spacing)))
    if weak_search_radius is None:
        weak_search_radius = max(2, int(round(0.20 * spacing)))
    if max_displacement is None:
        max_displacement = 0.30 * spacing
    if reference_subtract_box is None:
        reference_subtract_box = max(4, int(round(0.22 * spacing)))

    inside = inside_image(ideal, bright.shape,
                          margin=weak_fit_box + weak_search_radius + 1)
    ideal, pairs = ideal[inside], pairs[inside]
    if len(ideal) == 0:
        raise RuntimeError("All complete pairs were removed by the edge filter.")

    residual = (remove_columns_gaussian(bright, ref, box=reference_subtract_box,
                                        image_mode="ADF")
                if subtract_reference else bright.copy())
    seeds = local_residual_maxima(residual, ideal, weak_search_radius)
    found, params = refine_gaussian(residual, seeds, box=weak_fit_box,
                                    image_mode="ADF", return_params=True)
    disp = found - ideal
    amp = params["amplitude"]
    ell = params["ellipticity"]
    good = (params["success"] & np.isfinite(amp) & (amp > 0)
            & np.isfinite(ell) & (ell < max_ellipticity)
            & (np.linalg.norm(disp, axis=1) < max_displacement))
    if good.any():
        med = np.nanmedian(amp[good])
        if np.isfinite(med) and med > 0:
            good &= amp > min_amplitude * med

    result = DisplacementField(
        ideal[:, 0], ideal[:, 1], disp[:, 0], disp[:, 1],
        per_site={"pair_indices": pairs, "seeds": seeds, "amplitude": amp,
                  "ellipticity": ell},
        diagnostics={"residual": residual, "pair_vector": np.asarray(pair_vector),
                     "reference_fraction": reference_fraction})
    return result.subset(good).with_sign(sign)


# ----------------------------------------------------------------------------
# References from already-measured positions
# ----------------------------------------------------------------------------
def displacement_from_sublattices(A, B, v1, v2, offset=(0.5, 0.5), max_frac=0.34):
    """Displacements from already measured A and B sublattice positions.

    Use this when trusted sub-pixel positions of both sublattices are available
    (from this package or any other detector). For each A column the ideal
    position ``A + offset[0] v1 + offset[1] v2`` is matched to the nearest
    measured B column within ``max_frac * cell``.

    Note that this reference extrapolates the average lattice; the
    complete-cage reference (:func:`displacement_from_cage`) is more robust to
    lattice distortion.

    Returns
    -------
    DisplacementField
        Anchored at the ideal positions.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    cell = 0.5 * (np.linalg.norm(v1) + np.linalg.norm(v2))
    ideal = predict_second_sublattice(A, v1, v2, offset)
    dist, idx = KDTree(B).query(ideal, k=1)
    good = dist < max_frac * cell
    found = B[idx]
    return DisplacementField(ideal[good, 0], ideal[good, 1],
                             found[good, 0] - ideal[good, 0],
                             found[good, 1] - ideal[good, 1])


def displacement_from_cage(polar_xy, ref_xy, k=4, max_frac=1.5):
    """Displacement of each polar column from the centroid of its ``k`` nearest
    reference columns.

    For a perovskite [100] projection with ``k = 4`` this is the B-column shift
    from its A cage, computed from measured positions only. Columns whose
    ``k``-th neighbour is farther than ``max_frac`` times the median such
    distance (edges, defects) are dropped.

    Returns
    -------
    DisplacementField
        Anchored at the cage centroids.
    """
    polar = np.asarray(polar_xy, dtype=float)
    ref = np.asarray(ref_xy, dtype=float)
    dist, idx = KDTree(ref).query(polar, k=k)
    cage = ref[idx].mean(axis=1)
    good = dist[:, -1] < max_frac * np.median(dist[:, -1])
    return DisplacementField(cage[good, 0], cage[good, 1],
                             polar[good, 0] - cage[good, 0],
                             polar[good, 1] - cage[good, 1])


def displacement_from_offset(ref_xy, polar_xy, ideal_offset_px, max_dist=None):
    """Displacement from a fixed basis offset (dumbbell / wurtzite-type model).

    The ideal polar position is ``ref + ideal_offset_px`` for every reference
    column; the nearest measured polar column is matched and
    ``r_polar - (r_ref + offset)`` returned. For a wurtzite projection with the
    ``c`` axis along ``+y`` the ideal cation-to-anion offset is
    ``(0, u_ref * c / sampling)`` with ``u_ref = 3/8``.

    Parameters
    ----------
    ref_xy, polar_xy : array_like
        ``(N, 2)`` and ``(M, 2)`` positions.
    ideal_offset_px : array_like
        Ideal offset in pixels.
    max_dist : float, optional
        Maximum match distance; default half the offset length.

    Returns
    -------
    DisplacementField
        Anchored at the ideal positions.
    """
    ref = np.asarray(ref_xy, dtype=float)
    polar = np.asarray(polar_xy, dtype=float)
    off = np.asarray(ideal_offset_px, dtype=float)
    ideal = ref + off
    dist, idx = KDTree(polar).query(ideal, k=1)
    if max_dist is None:
        max_dist = 0.5 * np.linalg.norm(off)
    good = dist < max_dist
    found = polar[idx]
    return DisplacementField(ideal[good, 0], ideal[good, 1],
                             found[good, 0] - ideal[good, 0],
                             found[good, 1] - ideal[good, 1])


# ----------------------------------------------------------------------------
# Displacement -> polarization
# ----------------------------------------------------------------------------
def displacement_to_polarization(u_pm, z_star, cell_volume_nm3):
    """Semi-quantitative polarization from a single-sublattice displacement.

    .. math:: P = \\frac{e\\, Z^*\\, u}{\\Omega}

    Parameters
    ----------
    u_pm : array_like
        Displacement in pm.
    z_star : float
        Born effective charge of the displaced sublattice in units of ``e``
        (e.g. about +7.1 for Ti in PbTiO3; the nominal ionic charge is +4).
    cell_volume_nm3 : float
        Unit-cell volume in nm^3 (e.g. ``a * a * c``).

    Returns
    -------
    numpy.ndarray
        Polarization in µC/cm^2.

    Notes
    -----
    A rigorous polarization sums ``Z*_k u_k`` over every ion with anisotropic
    Born-charge tensors; oxygen positions are not measured in HAADF images.
    Report the measured displacement (pm) as the primary quantity.
    """
    e = 1.602176634e-19                          # C
    u_m = np.asarray(u_pm, dtype=float) * 1e-12  # pm -> m
    omega_m3 = float(cell_volume_nm3) * 1e-27    # nm^3 -> m^3
    return e * z_star * u_m / omega_m3 * 100.0   # C/m^2 -> µC/cm^2
