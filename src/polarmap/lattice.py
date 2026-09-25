"""Lattice vectors and the validated nearest-neighbour lattice graph.

Two ingredients are shared by the displacement and strain workflows:

1. **Lattice vectors** ``v1, v2``: the two shortest, non-parallel translations
   of a sublattice, estimated from the columns themselves
   (:func:`estimate_lattice_vectors`) or, for a pair of columns along one
   direction, :func:`estimate_pair_vector`.

2. **The lattice graph** (:class:`LatticeGraph`): for every column, its
   neighbour along ``+v1``, ``-v1``, ``+v2`` and ``-v2``. Neighbours are chosen
   by the smallest *transverse* offset from the ideal bond direction (robust to
   shear and scan jitter), and every bond is validated by a loop-closure test
   (:func:`find_bad_edges_by_closure`): walking ``+v1`` then ``+v2`` must reach
   the same column as walking ``+v2`` then ``+v1``. Integer lattice indices can
   then be propagated through the validated bonds by breadth-first search
   (:func:`bfs_lattice_indices`), which avoids the "phase slips" of assigning
   indices by rounding distances from a distant origin.
"""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.spatial import KDTree

__all__ = [
    "DIRECTION_NAMES",
    "estimate_lattice_vectors",
    "estimate_pair_vector",
    "LatticeGraph",
    "build_lattice_graph",
    "find_bad_edges_by_closure",
    "bfs_lattice_indices",
]

#: Order of the four signed bond directions used throughout the graph.
DIRECTION_NAMES = ("+v1", "-v1", "+v2", "-v2")

# Integer index step for each signed direction, in (n1, n2) lattice units.
_INDEX_STEP = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])


# ----------------------------------------------------------------------------
# Lattice-vector estimation
# ----------------------------------------------------------------------------
def _angdiff(a, b):
    """Smallest difference between two angles modulo pi, in (-pi/2, pi/2]."""
    return np.mod(a - b + np.pi / 2, np.pi) - np.pi / 2


def _robust_mean_vector(vecs, iters=3):
    v = vecs.mean(axis=0)
    for _ in range(iters):
        d = np.hypot(vecs[:, 0] - v[0], vecs[:, 1] - v[1])
        med = np.median(d)
        keep = d <= med + 2 * (np.median(np.abs(d - med)) + 1e-9)
        if keep.sum() < 3:
            break
        v = vecs[keep].mean(axis=0)
    return v


def _axial_mean(shell, angles, direction, width_deg=15.0):
    """Robust mean of the shell vectors within ``width_deg`` of an axis.

    ``+v`` and ``-v`` belong to the same axis, so every selected vector is
    first oriented along ``direction`` before averaging. Without this step,
    bonds lying almost exactly along the fold line of the half-plane (e.g. an
    x-aligned lattice with position noise) are split between ``+v`` and ``-v``
    and average to nearly zero.
    """
    sel = np.abs(_angdiff(angles, direction)) < np.deg2rad(width_deg)
    vecs = shell[sel].copy()
    u = np.array([np.cos(direction), np.sin(direction)])
    vecs[vecs @ u < 0] *= -1
    return _robust_mean_vector(vecs)


def estimate_lattice_vectors(points, n_neighbors=8, tol=0.3):
    """Estimate the two primitive lattice vectors of a near-periodic point set.

    The method uses nearest-neighbour statistics only, so no zone-axis
    information is needed:

    1. Collect the displacement vectors from every point to its
       ``n_neighbors`` nearest neighbours.
    2. Keep the first coordination shell: vectors whose length is within
       ``tol`` of the characteristic nearest-neighbour spacing (median of the
       shortest 40 % of all vectors).
    3. Treat ``+v`` and ``-v`` as the same axis: histogram the bond angles
       modulo 180 degrees in 5 degree bins and take the most populated axis.
    4. Take the most populated axis at least 25 degrees away from the first as
       the second axis.
    5. Each vector is the robust (iteratively outlier-trimmed) mean of the
       shell vectors within 15 degrees of its axis, after orienting them all
       the same way along that axis.
    6. The two vectors are ordered and oriented deterministically: ``v1`` is
       the one closer to the image ``x`` axis, pointing right (``v1[0] > 0``);
       ``v2`` points down (``v2[1] > 0``).

    Parameters
    ----------
    points : array_like
        ``(N, 2)`` column positions in pixels.
    n_neighbors : int, default 8
        Neighbours per point used for the statistics.
    tol : float, default 0.3
        Relative width of the first coordination shell.

    Returns
    -------
    v1, v2 : numpy.ndarray
        Lattice vectors in pixels (see step 6 for their order and signs).

    Warns
    -----
    UserWarning
        If no second direction is found in the first shell (e.g. a strongly
        rectangular projection with an aspect ratio above ``1 + tol``), in
        which case ``v2`` falls back to ``v1`` rotated by 90 degrees. Supply the
        vectors explicitly for such lattices.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 4:
        raise ValueError("need an (N, 2) array with at least 4 points")
    k = min(n_neighbors + 1, len(pts))
    _, idx = KDTree(pts).query(pts, k=k)
    vecs = (pts[idx[:, 1:]] - pts[:, None, :]).reshape(-1, 2)
    r = np.hypot(vecs[:, 0], vecs[:, 1])

    a = np.median(r[r <= np.percentile(r, 40)])          # shortest spacing
    shell = vecs[np.abs(r - a) <= tol * a]
    ang = np.mod(np.arctan2(shell[:, 1], shell[:, 0]), np.pi)   # axis, [0, pi)

    edges = np.linspace(0, np.pi, 37)                    # 5-degree bins
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist, _ = np.histogram(ang, bins=edges)
    a1 = centers[np.argmax(hist)]
    v1 = _axial_mean(shell, ang, a1)

    far = np.abs(_angdiff(ang, a1)) > np.deg2rad(25)
    if far.sum() >= 3:
        hist2, _ = np.histogram(ang[far], bins=edges)
        a2 = centers[np.argmax(hist2)]
        v2 = _axial_mean(shell[far], ang[far], a2)
    else:
        warnings.warn(
            "only one lattice direction found in the first coordination shell; "
            "v2 is set perpendicular to v1 with the same length. Pass the "
            "lattice vectors explicitly for strongly rectangular lattices.",
            stacklevel=2)
        v2 = np.array([-v1[1], v1[0]])

    # Deterministic order and signs: v1 closest to +x, v2 pointing down.
    if abs(v1[0]) / np.linalg.norm(v1) < abs(v2[0]) / np.linalg.norm(v2):
        v1, v2 = v2, v1
    if v1[0] < 0:
        v1 = -v1
    if v2[1] < 0:
        v2 = -v2
    return v1, v2


def estimate_pair_vector(reference_xy, axis_hint=(0.0, 1.0), k_neighbors=12,
                         angle_tolerance_deg=25.0):
    """Estimate the nearest reference-to-reference translation along an axis.

    For each reference column the nearest neighbour lying within
    ``angle_tolerance_deg`` of ``axis_hint`` (and in the positive half-plane of
    it) is selected; the component-wise median of those vectors, after
    rejecting outliers, is returned. A rough hint is therefore sufficient.

    Parameters
    ----------
    reference_xy : array_like
        ``(N, 2)`` reference column positions.
    axis_hint : (float, float), default (0, 1)
        Approximate direction in image coordinates (``x`` right, ``y`` down).
        ``(0, 1)`` means "the next reference column is below". The sign matters
        only when the reference fraction of a pair model differs from 0.5.
    k_neighbors : int, default 12
        Neighbours examined per column.
    angle_tolerance_deg : float, default 25
        Accepted angular deviation from ``axis_hint``.

    Returns
    -------
    numpy.ndarray
        The pair vector ``(dx, dy)`` in pixels.
    """
    ref = np.asarray(reference_xy, dtype=float)
    axis = np.asarray(axis_hint, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("axis_hint cannot be the zero vector")
    axis = axis / norm
    if len(ref) < 2:
        raise ValueError("need at least two reference columns")

    k = min(k_neighbors + 1, len(ref))
    _, indices = KDTree(ref).query(ref, k=k)
    indices = np.atleast_2d(indices)
    cos_min = np.cos(np.deg2rad(angle_tolerance_deg))
    selected = []
    for i, point in enumerate(ref):
        vectors = ref[np.atleast_1d(indices[i])[1:]] - point
        lengths = np.linalg.norm(vectors, axis=1)
        projection = vectors @ axis
        cosine = np.divide(projection, lengths,
                           out=np.full_like(projection, -np.inf),
                           where=lengths > 0)
        good = (projection > 0) & (cosine >= cos_min)
        if np.any(good):
            selected.append(vectors[good][np.argmin(lengths[good])])

    selected = np.asarray(selected, dtype=float)
    if len(selected) < 3:
        raise RuntimeError(
            "could not estimate the pair vector; adjust axis_hint or increase "
            "angle_tolerance_deg")
    median_vector = np.median(selected, axis=0)
    deviation = np.linalg.norm(selected - median_vector, axis=1)
    mad = np.median(np.abs(deviation - np.median(deviation)))
    limit = max(2.5 * mad, 0.20 * np.linalg.norm(median_vector), 1.0)
    return np.median(selected[deviation <= limit], axis=0)


# ----------------------------------------------------------------------------
# Lattice graph
# ----------------------------------------------------------------------------
@dataclass
class LatticeGraph:
    """Nearest neighbours of every column along the four signed lattice bonds.

    Build it with :func:`build_lattice_graph`.

    Attributes
    ----------
    positions : numpy.ndarray
        ``(N, 2)`` column positions ``(x, y)`` in pixels.
    directions : numpy.ndarray
        ``(4, 2)`` bond directions, ordered as :data:`DIRECTION_NAMES`:
        ``+v1, -v1, +v2, -v2``.
    neighbor_idx : numpy.ndarray
        ``(N, 4)`` index of the matched neighbour in each direction, ``-1`` if
        none was found.
    neighbor_vec : numpy.ndarray
        ``(N, 4, 2)`` bond vectors ``positions[neighbour] - positions[i]``
        (NaN where no neighbour was found).
    bad_edges : numpy.ndarray
        ``(N, 4)`` boolean mask of bonds rejected by the loop-closure test.
    """

    positions: np.ndarray
    directions: np.ndarray
    neighbor_idx: np.ndarray
    neighbor_vec: np.ndarray
    bad_edges: np.ndarray

    @property
    def n_atoms(self):
        """Number of columns in the graph."""
        return len(self.positions)

    @property
    def v1(self):
        """Lattice vector used for the ``±v1`` directions."""
        return self.directions[0]

    @property
    def v2(self):
        """Lattice vector used for the ``±v2`` directions."""
        return self.directions[2]

    @property
    def clean_idx(self):
        """``neighbor_idx`` with loop-closure-rejected bonds set to ``-1``."""
        idx = self.neighbor_idx.copy()
        idx[self.bad_edges] = -1
        return idx

    def direction_pair(self, vector):
        """Indices of the graph directions parallel and antiparallel to ``vector``.

        Parameters
        ----------
        vector : array_like
            Any vector (only its direction matters), e.g. ``v2`` for the
            out-of-plane direction.

        Returns
        -------
        plus, minus : int
            Column indices into ``neighbor_idx`` for the most parallel and the
            most antiparallel graph direction.
        """
        u = np.asarray(vector, dtype=float)
        u = u / np.linalg.norm(u)
        unit = self.directions / np.linalg.norm(self.directions, axis=1)[:, None]
        cosines = unit @ u
        return int(np.argmax(cosines)), int(np.argmin(cosines))

    def describe_column(self, i):
        """Explain, in words, which bonds of column ``i`` are used and why not.

        Use it to inspect a column with a missing or suspicious value: the
        cause is usually a missing or rejected bond, not a failed Gaussian fit.
        """
        lines = [f"column {i} at ({self.positions[i, 0]:.1f}, "
                 f"{self.positions[i, 1]:.1f}) px:"]
        for d, name in enumerate(DIRECTION_NAMES):
            j = self.neighbor_idx[i, d]
            if j < 0:
                lines.append(f"  {name}: no neighbour inside the search window")
            elif self.bad_edges[i, d]:
                lines.append(f"  {name}: column {j}, rejected by loop closure")
            else:
                length = np.linalg.norm(self.neighbor_vec[i, d])
                lines.append(f"  {name}: column {j}, bond length {length:.2f} px, used")
        return "\n".join(lines)

    def summary(self):
        """Counts describing bond matching and validation, as a dict."""
        found = self.neighbor_idx >= 0
        n_found = int(found.sum())
        return dict(
            n_atoms=self.n_atoms,
            n_bonds_possible=int(found.size),
            n_bonds_found=n_found,
            n_bonds_rejected=int(self.bad_edges.sum()),
            bonds_found_per_direction={
                name: int(found[:, d].sum())
                for d, name in enumerate(DIRECTION_NAMES)},
            n_atoms_missing_a_direction=int(np.any(~found, axis=1).sum()),
            n_atoms_with_rejected_bond=int(np.any(self.bad_edges, axis=1).sum()),
        )

    def report(self):
        """Human-readable summary of :meth:`summary`."""
        s = self.summary()
        found, possible = s["n_bonds_found"], s["n_bonds_possible"]
        lines = [
            f"Lattice graph: {s['n_atoms']} columns",
            f"  bonds found: {found} / {possible} ({100 * found / max(possible, 1):.1f} %)",
        ]
        for name, count in s["bonds_found_per_direction"].items():
            lines.append(f"    {name}: {count} / {s['n_atoms']}")
        lines.append(
            f"  bonds rejected by loop closure: {s['n_bonds_rejected']} / {found} "
            f"({100 * s['n_bonds_rejected'] / max(found, 1):.1f} %)")
        lines.append(f"  columns missing >= 1 direction: "
                     f"{s['n_atoms_missing_a_direction']}")
        return "\n".join(lines)


def build_lattice_graph(positions, v1, v2, low_frac=0.5, high_frac=1.5,
                        max_perp_frac=0.5, search_radius_factor=2.2,
                        check_closure=True):
    """Match every column to its neighbours along ``±v1`` and ``±v2``.

    For each column and each signed direction ``d`` (unit vector ``u``,
    length ``L``):

    1. candidates are the columns within ``search_radius_factor * max|v|``
       whose projection onto ``u`` lies in ``[low_frac * L, high_frac * L]``
       and whose perpendicular distance to the bond line is at most
       ``max_perp_frac * L``;
    2. among them, the one with the **smallest perpendicular distance** to the
       bond line is chosen.

    Choosing by perpendicular offset rather than by the smallest projected
    distance avoids picking a column from the adjacent row when the lattice is
    locally sheared or the scan jitters. The perpendicular limit (half a
    spacing, i.e. halfway to the adjacent row) prevents a *diagonal* neighbour
    from being accepted when the true neighbour is missing (image border,
    vacancy, undetected column); such a bond would otherwise be rejected later
    by the loop-closure test, together with the valid bonds around it.

    Parameters
    ----------
    positions : array_like
        ``(N, 2)`` refined column positions in pixels.
    v1, v2 : array_like
        Lattice vectors in pixels (e.g. from :func:`estimate_lattice_vectors`).
    low_frac, high_frac : float, default 0.5 and 1.5
        Accepted range of the projected bond length, relative to ``|v|``.
    max_perp_frac : float, default 0.5
        Maximum perpendicular offset from the bond line, relative to ``|v|``.
    search_radius_factor : float, default 2.2
        Candidate search radius relative to the longer lattice vector.
    check_closure : bool, default True
        Run :func:`find_bad_edges_by_closure`; if ``False``, no bond is marked
        as bad.

    Returns
    -------
    LatticeGraph
    """
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError(f"positions must have shape (N, 2), got {pos.shape}")
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    directions = np.array([v1, -v1, v2, -v2])
    lengths = np.linalg.norm(directions, axis=1)
    if np.any(lengths == 0) or not np.all(np.isfinite(lengths)):
        raise ValueError("lattice vectors must be finite and non-zero")
    if abs(v1[0] * v2[1] - v1[1] * v2[0]) < 1e-6 * lengths[0] * lengths[2]:
        raise ValueError("v1 and v2 must not be parallel")
    units = directions / lengths[:, None]

    n = len(pos)
    neighbor_idx = np.full((n, 4), -1, dtype=int)
    neighbor_vec = np.full((n, 4, 2), np.nan)
    if n:
        tree = KDTree(pos)
        search_r = search_radius_factor * lengths.max()
        for i, p in enumerate(pos):
            cand = np.array([j for j in tree.query_ball_point(p, r=search_r)
                             if j != i], dtype=int)
            if cand.size == 0:
                continue
            rel = pos[cand] - p
            for d in range(4):
                proj = rel @ units[d]
                perp = np.abs(rel @ np.array([-units[d, 1], units[d, 0]]))
                in_range = ((proj >= low_frac * lengths[d])
                            & (proj <= high_frac * lengths[d])
                            & (perp <= max_perp_frac * lengths[d]))
                if not np.any(in_range):
                    continue
                best = np.argmin(np.where(in_range, perp, np.inf))
                neighbor_idx[i, d] = cand[best]
                neighbor_vec[i, d] = rel[best]

    bad = (find_bad_edges_by_closure(neighbor_idx) if check_closure
           else np.zeros_like(neighbor_idx, dtype=bool))
    return LatticeGraph(pos, directions, neighbor_idx, neighbor_vec, bad)


def find_bad_edges_by_closure(neighbor_idx):
    """Flag bonds that violate loop closure on an elementary lattice cell.

    For a column ``i`` and each pair of directions ``(a, b)`` in
    ``{(+v1, +v2), (+v1, -v2), (-v1, +v2), (-v1, -v2)}``, the column reached by
    walking ``a`` then ``b`` must be the same as by walking ``b`` then ``a``.
    If both paths exist but end on different columns, at least one of the four
    bonds of that cell is wrong, so all four are flagged. The idea is the same
    as the residue (branch-cut) test used in two-dimensional phase unwrapping:
    inconsistent loops are cut out before anything is propagated through them.

    Parameters
    ----------
    neighbor_idx : numpy.ndarray
        ``(N, 4)`` neighbour indices ordered as :data:`DIRECTION_NAMES`
        (``-1`` = no neighbour).

    Returns
    -------
    numpy.ndarray
        ``(N, 4)`` boolean mask of rejected bonds.
    """
    idx = np.asarray(neighbor_idx)
    bad = np.zeros(idx.shape, dtype=bool)
    for da, db in ((0, 2), (0, 3), (1, 2), (1, 3)):
        for i in range(idx.shape[0]):
            j, j2 = idx[i, da], idx[i, db]
            if j < 0 or j2 < 0:
                continue
            k1, k2 = idx[j, db], idx[j2, da]
            if k1 < 0 or k2 < 0:
                continue
            if k1 != k2:
                bad[i, da] = bad[i, db] = True
                bad[j, db] = True
                bad[j2, da] = True
    return bad


def bfs_lattice_indices(neighbor_idx, anchor):
    """Assign integer lattice indices by breadth-first search from an anchor.

    The anchor column gets index ``(0, 0)``. Every column reached through a
    bond inherits the index of the column it was reached from, plus or minus
    one along the bond direction (``+v1`` adds ``(1, 0)``, ``-v2`` adds
    ``(0, -1)``, ...). Pass validated bonds (:attr:`LatticeGraph.clean_idx`)
    so that inconsistent bonds are never followed.

    Because each index is defined relative to an already-indexed neighbour, a
    local ambiguity cannot flip the indices of the whole region beyond it, as
    happens when indices are obtained by rounding ``(r - r0) / v`` from a
    distant origin once the accumulated drift exceeds half a cell.

    Parameters
    ----------
    neighbor_idx : numpy.ndarray
        ``(N, 4)`` neighbour indices (``-1`` = no bond).
    anchor : int
        Index of the starting column.

    Returns
    -------
    indices : numpy.ndarray
        ``(N, 2)`` float array of integer indices ``(n1, n2)``; NaN for columns
        not connected to the anchor.
    reached : numpy.ndarray
        ``(N,)`` boolean mask of connected columns.
    """
    idx = np.asarray(neighbor_idx)
    n = idx.shape[0]
    if not 0 <= anchor < n:
        raise IndexError(f"anchor {anchor} out of range for {n} columns")
    indices = np.full((n, 2), np.nan)
    reached = np.zeros(n, dtype=bool)
    indices[anchor] = 0.0
    reached[anchor] = True
    queue = deque([anchor])
    while queue:
        i = queue.popleft()
        for d in range(4):
            j = idx[i, d]
            if j < 0 or reached[j]:
                continue
            indices[j] = indices[i] + _INDEX_STEP[d]
            reached[j] = True
            queue.append(j)
    return indices, reached
