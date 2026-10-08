"""Real-space strain mapping from atomic-column positions.

Two complementary methods share the validated lattice graph of
:mod:`polarmap.lattice`:

**Method 1 -- projection strain against an external reference**
(:func:`projection_strain`). For every column, the local spacing along a
chosen lattice direction is the projection of its validated bonds onto that
direction (symmetric average of the ``+`` and ``-`` bonds when both exist).
The engineering strain is ``(d - d0) / d0`` with ``d0`` an external reference
spacing, e.g. a bulk lattice parameter. No unstrained region is needed in the
field of view, but the result inherits any error of the pixel calibration.

**Method 2 -- full 2-D strain and rotation tensor against an internal
reference** (:func:`fit_reference_lattice` then :func:`tensor_strain`).

* A reference lattice (origin, **a**, **b**) is fitted by least squares to the
  columns of a user-selected region assumed unstrained, using integer indices
  propagated by breadth-first search over validated bonds. The worst-fitting
  20 % of the reference columns are rejected before the final fit.
* For every column *i*, the local deformation gradient :math:`F_i` is the
  least-squares solution of :math:`\\mathbf{m}_d \\approx F_i\\,\\mathbf{e}_d`
  over its validated bonds, where :math:`\\mathbf{m}_d` is the measured bond
  vector and :math:`\\mathbf{e}_d \\in \\{\\pm\\mathbf{a}, \\pm\\mathbf{b}\\}` the
  corresponding reference vector. At least one bond along each axis is
  required.
* Strain and rigid-body rotation are the symmetric and antisymmetric parts:
  :math:`\\varepsilon = \\tfrac12(F + F^\\top) - I`,
  :math:`\\omega = \\tfrac12(F_{yx} - F_{xy})`.

Because :math:`F_i` depends only on the bonds of column *i*, an error in one
column cannot spread to its neighbours, and a small bias of the fitted
reference vectors produces a *uniform* offset instead of an error that grows
with the distance from the reference region (as happens when displacements
from a rigidly extrapolated lattice are interpolated and differentiated). The
strain is a ratio of pixel distances and is therefore independent of the pixel
calibration.

Axes and signs
--------------
Tensor components are expressed in image axes: ``x`` to the right, ``y``
**down**. ``exx`` and ``eyy`` are unchanged in a y-up frame, whereas ``exy``
and ``omega`` change sign; :meth:`TensorStrain.in_frame` returns the components
in a y-up Cartesian frame or rotated into the lattice frame. In image axes a
positive ``omega`` is a clockwise rotation as seen on the screen.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.interpolate import griddata

from .displacement import DisplacementField
from .geometry import inside_image
from .lattice import LatticeGraph, bfs_lattice_indices
from .statistics import local_outlier_mask, mad_outlier_mask

__all__ = [
    "ProjectionStrain",
    "projection_strain",
    "ReferenceLattice",
    "fit_reference_lattice",
    "rigid_lattice_displacement",
    "TensorStrain",
    "deformation_gradients",
    "strain_from_deformation_gradient",
    "tensor_strain",
    "rotate_strain",
    "interpolate_to_grid",
    "binned_profile",
    "line_profile",
]


# ============================================================================
# Method 1: projection strain against an external reference spacing
# ============================================================================
@dataclass
class ProjectionStrain:
    """Result of :func:`projection_strain` (Method 1).

    Attributes
    ----------
    positions : numpy.ndarray
        ``(N, 2)`` column positions.
    strain : numpy.ndarray
        ``(N,)`` engineering strain; NaN where no valid value exists.
    spacing_px : numpy.ndarray
        ``(N,)`` local spacing along the direction, in pixels (NaN if none).
    direction : numpy.ndarray
        Unit vector along which the spacing was measured.
    d0_px : float or numpy.ndarray
        Reference spacing(s) in pixels.
    status : numpy.ndarray
        ``(N,)`` integer code explaining each column; see :attr:`STATUS`.
    """

    positions: np.ndarray
    strain: np.ndarray
    spacing_px: np.ndarray
    direction: np.ndarray
    d0_px: object
    status: np.ndarray

    #: Meaning of the codes in :attr:`status`.
    STATUS = {0: "valid", 1: "no neighbour along the direction",
              2: "bonds rejected by loop closure", 3: "within edge margin",
              4: "robust outlier"}

    @property
    def valid(self):
        """Boolean mask of columns with a strain value."""
        return self.status == 0

    def summary(self):
        """Number of columns per status, as a dict keyed by description."""
        return {text: int(np.sum(self.status == code))
                for code, text in self.STATUS.items()}

    def report(self):
        """Human-readable summary."""
        n = len(self.status)
        lines = [f"Projection strain (Method 1): {n} columns"]
        for text, count in self.summary().items():
            lines.append(f"  {text}: {count} ({100 * count / max(n, 1):.1f} %)")
        vals = self.strain[self.valid]
        if vals.size:
            lines.append(f"  strain: mean {100 * vals.mean():+.2f} %, "
                         f"std {100 * vals.std():.2f} %")
        return "\n".join(lines)


def projection_strain(graph, direction, d0_px, *, image_shape=None, edge=20.0,
                      outlier_nsigma=5.0, outlier_scope="local"):
    """Method 1: strain along one lattice direction relative to ``d0``.

    For each column, with ``u`` the unit vector along ``direction``:

    * ``d+`` is the projection onto ``u`` of the validated bond along the
      graph direction most parallel to ``u``, and ``d-`` that of the bond along
      the antiparallel direction (sign-corrected);
    * the local spacing is ``(d+ + d-) / 2`` when both exist, otherwise the one
      available;
    * the strain is ``(d - d0) / d0``.

    Columns are then excluded (and counted in ``status``) if they lie within
    ``edge`` pixels of the border or if they are robust outliers: by default
    (``outlier_scope="local"``) a column is an outlier when its strain deviates
    from the median of its 8 nearest neighbours by more than
    ``outlier_nsigma`` robust standard deviations of these residuals
    (:func:`polarmap.statistics.local_outlier_mask`). Nothing is removed by a
    fixed percentile.

    .. note::
       ``outlier_scope="global"`` compares every column with the median of the
       whole map. That is only appropriate for a single, homogeneous region:
       in a heterostructure it flags a strained layer as a whole whenever it
       covers less than about half of the columns.

    Parameters
    ----------
    graph : LatticeGraph
        Validated lattice graph.
    direction : array_like
        Direction of interest, e.g. ``graph.v2`` for the out-of-plane lattice
        vector; only its orientation matters.
    d0_px : float or array_like
        Reference spacing in pixels (``d0_nm / sampling``). An ``(N,)`` array
        assigns a reference to each column, e.g. a different bulk value in each
        layer of a heterostructure.
    image_shape : tuple, optional
        ``(height, width)``; needed for the edge exclusion.
    edge : float, default 20
        Edge margin in pixels (ignored without ``image_shape``).
    outlier_nsigma : float or None, default 5
        Robust outlier threshold; ``None`` disables the test.
    outlier_scope : {"local", "global"}, default "local"
        Neighbourhood-based or whole-map outlier test (see the note above).

    Returns
    -------
    ProjectionStrain
    """
    if outlier_scope not in ("local", "global"):
        raise ValueError("outlier_scope must be 'local' or 'global'")
    if not isinstance(graph, LatticeGraph):
        raise TypeError("graph must be a LatticeGraph (see build_lattice_graph)")
    u = np.asarray(direction, dtype=float)
    u = u / np.linalg.norm(u)
    d0 = np.asarray(d0_px, dtype=float)
    if np.any(~np.isfinite(d0)) or np.any(d0 <= 0):
        raise ValueError("d0_px must be positive and finite")
    if d0.ndim == 1 and len(d0) != graph.n_atoms:
        raise ValueError("an array d0_px needs one value per column")

    plus, minus = graph.direction_pair(u)
    clean = graph.clean_idx
    has_p, has_m = clean[:, plus] >= 0, clean[:, minus] >= 0
    with np.errstate(invalid="ignore"):
        d_plus = np.where(has_p, graph.neighbor_vec[:, plus] @ u, np.nan)
        d_minus = np.where(has_m, -(graph.neighbor_vec[:, minus] @ u), np.nan)
    spacing = np.where(has_p & has_m, 0.5 * (d_plus + d_minus),
                       np.where(has_p, d_plus, d_minus))

    status = np.zeros(graph.n_atoms, dtype=int)
    none = ~(has_p | has_m)
    raw_none = (graph.neighbor_idx[:, plus] < 0) & (graph.neighbor_idx[:, minus] < 0)
    status[none & raw_none] = 1
    status[none & ~raw_none] = 2
    strain = (spacing - d0) / d0

    if image_shape is not None:
        at_edge = ~inside_image(graph.positions, image_shape, margin=edge)
        status[(status == 0) & at_edge] = 3
    if outlier_nsigma is not None:
        values = np.where(status == 0, strain, np.nan)
        if outlier_scope == "local":
            outliers = local_outlier_mask(graph.positions, values, outlier_nsigma)
        else:
            outliers = mad_outlier_mask(values, outlier_nsigma, two_sided=True,
                                        normal_consistent=True)
        status[outliers] = 4
    strain = np.where(status == 0, strain, np.nan)
    spacing = np.where(np.isin(status, (1, 2)), np.nan, spacing)
    return ProjectionStrain(graph.positions, strain, spacing, u,
                            d0 if d0.ndim else float(d0), status)


# ============================================================================
# Method 2: reference lattice + per-column deformation gradient
# ============================================================================
@dataclass
class ReferenceLattice:
    """Reference lattice fitted in an (assumed) unstrained region.

    Produced by :func:`fit_reference_lattice`.

    Attributes
    ----------
    origin : numpy.ndarray
        Fitted position of the lattice site with index ``(0, 0)``.
    a, b : numpy.ndarray
        Fitted lattice vectors along the graph's ``v1`` and ``v2``.
    indices : numpy.ndarray
        ``(N, 2)`` integer lattice indices from BFS (NaN if unreached).
    reached : numpy.ndarray
        ``(N,)`` columns connected to the anchor through validated bonds.
    anchor : int
        Index of the BFS starting column.
    reference_mask : numpy.ndarray
        ``(N,)`` columns inside the reference region.
    used_mask : numpy.ndarray
        ``(N,)`` reference columns used in the final fit.
    residuals : numpy.ndarray
        ``(N,)`` fit residual (px) of the reached reference columns, NaN
        elsewhere.
    initial_residual : dict
        Mean and maximum residual (px) in the reference region when the
        initial lattice vectors are used (a consistency check of the
        indexing).
    """

    origin: np.ndarray
    a: np.ndarray
    b: np.ndarray
    indices: np.ndarray
    reached: np.ndarray
    anchor: int
    reference_mask: np.ndarray
    used_mask: np.ndarray
    residuals: np.ndarray
    initial_residual: dict = field(default_factory=dict)

    @property
    def vectors(self):
        """``(2, 2)`` array with rows ``a`` and ``b``."""
        return np.vstack([self.a, self.b])

    def ideal_positions(self):
        """Positions of a perfect lattice rigidly extrapolated from the fit.

        ``origin + n1 a + n2 b`` for every reached column (NaN otherwise).
        Used for diagnostics only; see :func:`rigid_lattice_displacement`.
        """
        return self.origin + self.indices @ self.vectors

    def summary(self):
        """Fit statistics as a dict."""
        res = self.residuals[self.used_mask]
        return dict(n_reference=int(self.reference_mask.sum()),
                    n_reached=int(self.reached.sum()),
                    n_used=int(self.used_mask.sum()),
                    residual_mean_px=float(res.mean()) if res.size else np.nan,
                    residual_max_px=float(res.max()) if res.size else np.nan,
                    initial_residual=dict(self.initial_residual),
                    a=self.a.copy(), b=self.b.copy())

    def report(self, sampling=None):
        """Human-readable summary; lengths in nm if ``sampling`` is given."""
        s = self.summary()
        n = len(self.reached)
        lines = [
            "Reference lattice (Method 2)",
            f"  BFS reached {s['n_reached']} / {n} columns "
            f"({100 * s['n_reached'] / max(n, 1):.1f} %)",
            f"  reference columns used in final fit: {s['n_used']} / "
            f"{s['n_reference']}",
            f"  fit residual: mean {s['residual_mean_px']:.3f} px, "
            f"max {s['residual_max_px']:.3f} px",
        ]
        if self.initial_residual:
            lines.append(
                f"  residual with initial vectors: mean "
                f"{self.initial_residual['mean_px']:.3f} px, max "
                f"{self.initial_residual['max_px']:.3f} px")
        for name, vec in (("a", self.a), ("b", self.b)):
            length = np.linalg.norm(vec)
            extra = f" = {length * sampling:.5f} nm" if sampling else ""
            lines.append(f"  {name} = ({vec[0]:.3f}, {vec[1]:.3f}) px, "
                         f"|{name}| = {length:.4f} px{extra}")
        return "\n".join(lines)


def fit_reference_lattice(graph, reference_mask, *, anchor=None,
                          reject_fraction=0.2, min_atoms=9):
    """Fit origin and lattice vectors to the columns of a reference region.

    Stage A: integer indices ``(n1, n2)`` are propagated by breadth-first
    search over the validated bonds (:func:`~polarmap.lattice.bfs_lattice_indices`),
    starting from ``anchor``.

    Stage B: the linear model ``r_i = origin + n1_i a + n2_i b`` is fitted by
    least squares to the reached reference columns; the ``reject_fraction``
    with the largest residuals are discarded and the model is refitted.

    Parameters
    ----------
    graph : LatticeGraph
    reference_mask : array_like of bool
        ``(N,)`` columns inside the reference region (e.g. from
        :func:`polarmap.geometry.in_rectangle`).
    anchor : int, optional
        Starting column; default: the reference column closest to the
        centroid of the reference columns.
    reject_fraction : float, default 0.2
        Fraction of reference columns rejected before the final fit.
    min_atoms : int, default 9
        Minimum number of reference columns.

    Returns
    -------
    ReferenceLattice
    """
    ref_mask = np.asarray(reference_mask, dtype=bool)
    if ref_mask.shape != (graph.n_atoms,):
        raise ValueError("reference_mask must have one entry per column")
    if ref_mask.sum() < min_atoms:
        raise ValueError(
            f"only {int(ref_mask.sum())} columns in the reference region "
            f"(need >= {min_atoms}); enlarge or move the region")
    if not 0 <= reject_fraction < 1:
        raise ValueError("reject_fraction must be in [0, 1)")

    pos = graph.positions
    if anchor is None:
        ref_idx = np.flatnonzero(ref_mask)
        centre = pos[ref_idx].mean(axis=0)
        anchor = int(ref_idx[np.argmin(np.linalg.norm(pos[ref_idx] - centre,
                                                      axis=1))])
    indices, reached = bfs_lattice_indices(graph.clean_idx, anchor)

    fit_idx = np.flatnonzero(ref_mask & reached)
    if len(fit_idx) < min_atoms:
        raise ValueError(
            f"only {len(fit_idx)} reference columns are connected to the anchor "
            "through validated bonds; check the lattice graph")

    # Consistency check of the indexing with the initial vectors.
    initial = pos[anchor] + indices[fit_idx] @ np.vstack([graph.v1, graph.v2])
    init_res = np.linalg.norm(pos[fit_idx] - initial, axis=1)
    initial_residual = dict(mean_px=float(init_res.mean()),
                            max_px=float(init_res.max()))

    design = np.column_stack([np.ones(len(fit_idx)), indices[fit_idx]])
    params, *_ = np.linalg.lstsq(design, pos[fit_idx], rcond=None)
    res = np.linalg.norm(pos[fit_idx] - design @ params, axis=1)
    keep = res <= np.percentile(res, 100 * (1 - reject_fraction))
    if keep.sum() < min_atoms:
        warnings.warn("too few reference columns after outlier rejection; "
                      "using all connected reference columns", stacklevel=2)
        keep[:] = True
    params, *_ = np.linalg.lstsq(design[keep], pos[fit_idx][keep], rcond=None)

    residuals = np.full(graph.n_atoms, np.nan)
    residuals[fit_idx] = np.linalg.norm(pos[fit_idx] - design @ params, axis=1)
    used = np.zeros(graph.n_atoms, dtype=bool)
    used[fit_idx[keep]] = True
    return ReferenceLattice(origin=params[0], a=params[1], b=params[2],
                            indices=indices, reached=reached, anchor=anchor,
                            reference_mask=ref_mask, used_mask=used,
                            residuals=residuals,
                            initial_residual=initial_residual)


def rigid_lattice_displacement(graph, reference):
    """Displacement of every column from the rigidly extrapolated reference lattice.

    ``u = r - (origin + n1 a + n2 b)``. This is a *diagnostic*: any small
    mismatch between the fitted vectors and the true lattice far from the
    reference region makes ``|u|`` grow linearly with distance, so it must not
    be differentiated to obtain strain (use :func:`tensor_strain`).

    Parameters
    ----------
    graph : LatticeGraph
        The graph the reference was fitted on.
    reference : ReferenceLattice

    Returns
    -------
    DisplacementField
        Anchored at the ideal positions of the reached columns; ``per_site``
        contains ``index`` (row into the graph).
    """
    ideal = reference.ideal_positions()
    ok = reference.reached
    u = graph.positions[ok] - ideal[ok]
    return DisplacementField(ideal[ok, 0], ideal[ok, 1], u[:, 0], u[:, 1],
                             per_site={"index": np.flatnonzero(ok)})


@dataclass
class TensorStrain:
    """Per-column strain and rotation tensor (Method 2).

    Attributes
    ----------
    positions : numpy.ndarray
        ``(N, 2)`` column positions.
    F : numpy.ndarray
        ``(N, 2, 2)`` local deformation gradients (NaN where undefined).
    exx, eyy, exy : numpy.ndarray
        Strain components (dimensionless; multiply by 100 for %).
    omega : numpy.ndarray
        Rigid-body rotation (radians, small-angle).
    n_bonds : numpy.ndarray
        Number of validated bonds used per column.
    frame : str
        ``"image"`` (x right, y down), ``"cartesian"`` (x right, y up) or
        ``"rotated"`` (see :meth:`in_frame`).
    """

    positions: np.ndarray
    F: np.ndarray
    exx: np.ndarray
    eyy: np.ndarray
    exy: np.ndarray
    omega: np.ndarray
    n_bonds: np.ndarray
    frame: str = "image"

    @property
    def valid(self):
        """Columns with a defined tensor."""
        return np.isfinite(self.exx)

    def component(self, name):
        """Return ``exx``, ``eyy``, ``exy`` or ``omega`` by name."""
        if name not in ("exx", "eyy", "exy", "omega"):
            raise ValueError("name must be 'exx', 'eyy', 'exy' or 'omega'")
        return getattr(self, name)

    def in_frame(self, frame="cartesian", angle_deg=None):
        """Express the tensor in another frame.

        Parameters
        ----------
        frame : {"image", "cartesian", "rotated"}
            ``"cartesian"`` flips the y axis (``exy`` and ``omega`` change
            sign). ``"rotated"`` rotates the image-frame axes by ``angle_deg``
            (counter-clockwise in image coordinates, i.e. from ``+x`` towards
            ``+y``), e.g. to align ``x`` with the in-plane lattice direction.
        angle_deg : float, optional
            Required for ``"rotated"``.
        """
        if self.frame != "image":
            raise ValueError("convert from the image frame only")
        if frame == "image":
            return self
        if frame == "cartesian":
            return replace(self, exy=-self.exy, omega=-self.omega,
                           frame="cartesian")
        if frame == "rotated":
            if angle_deg is None:
                raise ValueError("angle_deg is required for frame='rotated'")
            exx, eyy, exy = rotate_strain(self.exx, self.eyy, self.exy,
                                          np.deg2rad(angle_deg))
            return replace(self, exx=exx, eyy=eyy, exy=exy, frame="rotated")
        raise ValueError("frame must be 'image', 'cartesian' or 'rotated'")

    def summary(self):
        """Mean and standard deviation of every component (dimensionless)."""
        out = {"n_valid": int(self.valid.sum()), "n_total": len(self.exx)}
        for name in ("exx", "eyy", "exy", "omega"):
            vals = self.component(name)[self.valid]
            out[name] = dict(mean=float(vals.mean()) if vals.size else np.nan,
                             std=float(vals.std()) if vals.size else np.nan)
        return out

    def report(self):
        """Human-readable summary (strain in %, rotation in degrees)."""
        s = self.summary()
        lines = [f"Tensor strain (Method 2, {self.frame} frame): "
                 f"{s['n_valid']} / {s['n_total']} columns with a tensor"]
        for name in ("exx", "eyy", "exy"):
            lines.append(f"  {name}: mean {100 * s[name]['mean']:+.2f} %, "
                         f"std {100 * s[name]['std']:.2f} %")
        lines.append(f"  omega: mean {np.degrees(s['omega']['mean']):+.3f} deg, "
                     f"std {np.degrees(s['omega']['std']):.3f} deg")
        return "\n".join(lines)


def deformation_gradients(graph, reference_vectors, mask=None):
    """Local deformation gradient of every column from its validated bonds.

    Solves, for each column *i*, the linear least-squares problem

    .. math:: \\min_{F_i} \\sum_d \\lVert \\mathbf{m}_{d} - F_i\\,\\mathbf{e}_{d}\\rVert^2

    over its validated bonds ``d``, where ``m_d`` is the measured bond vector
    and ``e_d`` the reference vector (``+a, -a, +b, -b`` for the four graph
    directions). The closed-form solution is
    :math:`F_i = (\\sum_d \\mathbf{m}_d \\mathbf{e}_d^\\top)
    (\\sum_d \\mathbf{e}_d \\mathbf{e}_d^\\top)^{-1}`, evaluated for all columns
    at once.

    Parameters
    ----------
    graph : LatticeGraph
    reference_vectors : array_like
        ``(2, 2)`` array with rows ``a`` and ``b`` (one reference lattice for all
        columns), or ``(N, 2, 2)`` to give each column its own reference, e.g.
        the bulk lattice of the layer it belongs to.
    mask : array_like of bool, optional
        Columns to evaluate (others are NaN).

    Returns
    -------
    F : numpy.ndarray
        ``(N, 2, 2)``; NaN for columns without at least one validated bond
        along each lattice direction.
    n_bonds : numpy.ndarray
        ``(N,)`` number of validated bonds used.
    """
    ref = np.asarray(reference_vectors, dtype=float)
    n = graph.n_atoms
    if ref.shape == (2, 2):
        ref = np.broadcast_to(ref, (n, 2, 2))
    elif ref.shape != (n, 2, 2):
        raise ValueError("reference_vectors must have shape (2, 2) or (N, 2, 2)")
    e = np.stack([ref[:, 0], -ref[:, 0], ref[:, 1], -ref[:, 1]], axis=1)  # (N,4,2)

    w = (graph.clean_idx >= 0).astype(float)                              # (N,4)
    if mask is not None:
        w *= np.asarray(mask, dtype=bool)[:, None]
    m = np.nan_to_num(graph.neighbor_vec)                                 # (N,4,2)
    n_bonds = w.sum(axis=1).astype(int)
    solvable = (w[:, :2].sum(axis=1) > 0) & (w[:, 2:].sum(axis=1) > 0)

    gram = np.einsum("nd,ndi,ndj->nij", w, e, e)           # sum_d e e^T
    cross = np.einsum("nd,ndi,ndj->nij", w, m, e)          # sum_d m e^T
    F = np.full((n, 2, 2), np.nan)
    if solvable.any():
        F[solvable] = cross[solvable] @ np.linalg.inv(gram[solvable])
    return F, n_bonds


def strain_from_deformation_gradient(F):
    """Strain components and rotation from deformation gradients.

    Parameters
    ----------
    F : array_like
        ``(..., 2, 2)`` deformation gradients.

    Returns
    -------
    exx, eyy, exy, omega : numpy.ndarray
        ``eps = (F + F^T) / 2 - I`` and ``omega = (F[1, 0] - F[0, 1]) / 2``.
    """
    F = np.asarray(F, dtype=float)
    exx = F[..., 0, 0] - 1.0
    eyy = F[..., 1, 1] - 1.0
    exy = 0.5 * (F[..., 0, 1] + F[..., 1, 0])
    omega = 0.5 * (F[..., 1, 0] - F[..., 0, 1])
    return exx, eyy, exy, omega


def tensor_strain(graph, reference, *, mask=None):
    """Method 2: per-column strain and rotation tensor.

    Parameters
    ----------
    graph : LatticeGraph
    reference : ReferenceLattice or array_like
        The fitted reference lattice, or reference vectors as accepted by
        :func:`deformation_gradients` (``(2, 2)`` or ``(N, 2, 2)``).
    mask : array_like of bool, optional
        Columns to evaluate, e.g. ``inside_image(graph.positions, shape,
        margin)`` to leave out the image border.

    Returns
    -------
    TensorStrain
        In the image frame.
    """
    vectors = reference.vectors if isinstance(reference, ReferenceLattice) else reference
    F, n_bonds = deformation_gradients(graph, vectors, mask=mask)
    exx, eyy, exy, omega = strain_from_deformation_gradient(F)
    return TensorStrain(graph.positions, F, exx, eyy, exy, omega, n_bonds)


def rotate_strain(exx, eyy, exy, angle):
    """Rotate in-plane strain components by ``angle`` (radians).

    Returns the components in axes rotated by ``angle`` from the original ones
    (``eps' = R eps R^T`` with ``R = [[c, s], [-s, c]]``). The trace
    ``exx + eyy`` is invariant.
    """
    c, s = np.cos(angle), np.sin(angle)
    exx = np.asarray(exx, dtype=float)
    eyy = np.asarray(eyy, dtype=float)
    exy = np.asarray(exy, dtype=float)
    exx_r = c * c * exx + 2 * c * s * exy + s * s * eyy
    eyy_r = s * s * exx - 2 * c * s * exy + c * c * eyy
    exy_r = (c * c - s * s) * exy + c * s * (eyy - exx)
    return exx_r, eyy_r, exy_r


# ============================================================================
# Visualisation helpers
# ============================================================================
def interpolate_to_grid(positions, values, step=2.0, extent=None,
                        method="linear"):
    """Interpolate per-column values onto a regular pixel grid (for display).

    The per-column values are the measurement; the grid only provides a
    continuous picture. Linear interpolation on the Delaunay triangulation
    (``scipy.interpolate.griddata``) introduces no new extrema; points outside
    the convex hull of the valid columns are NaN.

    Parameters
    ----------
    positions : array_like
        ``(N, 2)`` column positions.
    values : array_like
        ``(N,)`` values; NaNs are ignored.
    step : float, default 2
        Grid spacing in pixels.
    extent : (xmin, xmax, ymin, ymax), optional
        Grid limits in pixels; default: bounding box of the valid columns.
    method : {"linear", "nearest", "cubic"}, default "linear"

    Returns
    -------
    grid : numpy.ndarray
        ``(len(yi), len(xi))`` interpolated values.
    xi, yi : numpy.ndarray
        Grid coordinates in pixels.
    """
    pos = np.asarray(positions, dtype=float)
    vals = np.asarray(values, dtype=float)
    ok = np.isfinite(vals) & np.all(np.isfinite(pos), axis=1)
    if ok.sum() < 3:
        raise ValueError("need at least 3 finite values to interpolate")
    if extent is None:
        extent = (pos[ok, 0].min(), pos[ok, 0].max(),
                  pos[ok, 1].min(), pos[ok, 1].max())
    xi = np.arange(extent[0], extent[1] + step / 2, step, dtype=float)
    yi = np.arange(extent[2], extent[3] + step / 2, step, dtype=float)
    xx, yy = np.meshgrid(xi, yi)
    grid = griddata(pos[ok], vals[ok], (xx, yy), method=method)
    return grid, xi, yi


def binned_profile(positions, values, axis="y", bin_width=None, value_range=None):
    """Average per-column values in bands along one image axis.

    For a multilayer grown along ``y``, ``binned_profile(pos, eyy, axis="y")``
    gives the mean strain of each band of rows, with its spread, directly from
    the per-column measurements (no interpolation).

    Parameters
    ----------
    positions : array_like
        ``(N, 2)`` column positions (px).
    values : array_like
        ``(N,)`` values; NaNs are ignored.
    axis : {"x", "y"}, default "y"
        Coordinate along which the profile runs.
    bin_width : float, optional
        Band width in pixels; default: 50 bands over the data range.
    value_range : (float, float), optional
        Coordinate range (px); default: the data range.

    Returns
    -------
    dict
        ``center``, ``mean``, ``std``, ``sem`` (standard error) and ``count``
        per band (bands without data are NaN / 0).
    """
    pos = np.asarray(positions, dtype=float)
    vals = np.asarray(values, dtype=float)
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")
    coord = pos[:, 0] if axis == "x" else pos[:, 1]
    ok = np.isfinite(vals) & np.isfinite(coord)
    if not ok.any():
        raise ValueError("no finite values")
    lo, hi = value_range if value_range is not None else (coord[ok].min(),
                                                           coord[ok].max())
    if bin_width is None:
        bin_width = (hi - lo) / 50 if hi > lo else 1.0
    edges = np.arange(lo, hi + bin_width, bin_width)
    if len(edges) < 2:
        edges = np.array([lo, lo + bin_width])
    which = np.clip(np.digitize(coord[ok], edges) - 1, 0, len(edges) - 2)
    n_bins = len(edges) - 1
    count = np.bincount(which, minlength=n_bins)
    total = np.bincount(which, weights=vals[ok], minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
        # two-pass variance: numerically stable, unlike E[x^2] - E[x]^2
        dev2 = np.bincount(which, weights=(vals[ok] - mean[which]) ** 2,
                           minlength=n_bins)
        std = np.sqrt(dev2 / count)
        sem = std / np.sqrt(count)
    return dict(center=0.5 * (edges[:-1] + edges[1:]), mean=mean, std=std,
                sem=sem, count=count)


def line_profile(grid, xi, yi, axis="x", coordinate=None):
    """Extract a straight line profile from a gridded map.

    Parameters
    ----------
    grid, xi, yi
        Output of :func:`interpolate_to_grid`.
    axis : {"x", "y"}, default "x"
        ``"x"``: a vertical cut at fixed ``x = coordinate`` (profile along y);
        ``"y"``: a horizontal cut at fixed ``y = coordinate``.
    coordinate : float, optional
        Position of the cut in pixels; default the centre of the grid.

    Returns
    -------
    coords, profile : numpy.ndarray
        Coordinates along the cut (px) and the values.
    """
    grid = np.asarray(grid)
    if axis == "x":
        coordinate = xi.mean() if coordinate is None else coordinate
        col = int(np.clip(np.argmin(np.abs(xi - coordinate)), 0, grid.shape[1] - 1))
        return yi, grid[:, col]
    if axis == "y":
        coordinate = yi.mean() if coordinate is None else coordinate
        row = int(np.clip(np.argmin(np.abs(yi - coordinate)), 0, grid.shape[0] - 1))
        return xi, grid[row, :]
    raise ValueError("axis must be 'x' or 'y'")
