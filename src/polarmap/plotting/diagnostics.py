"""Diagnostic figures: check each step before interpreting a map."""

from __future__ import annotations

import numpy as np
import matplotlib.patches as mpatches
from scipy.spatial import KDTree

from ..lattice import DIRECTION_NAMES
from ._common import get_axes, show_image

__all__ = [
    "plot_columns",
    "plot_cages",
    "plot_pairs",
    "plot_intensity_split",
    "plot_neighbor_diagnostic",
    "plot_reference_region",
    "plot_lattice_planes",
    "plot_neighbor_cloud",
]


def _image_limits(ax, image):
    h, w = np.asarray(image).shape
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect("equal")


def plot_columns(image, positions, *, ax=None, color="r", s=10, marker="o",
                 title=None, label=None):
    """Overlay detected or refined column positions on the image.

    Check that every intended column carries exactly one marker and that the
    other sublattice is not picked up.
    """
    fig, ax = get_axes(ax)
    show_image(ax, image)
    pos = np.asarray(positions, dtype=float).reshape(-1, 2)
    ax.scatter(pos[:, 0], pos[:, 1], s=s, facecolors="none", edgecolors=color,
               marker=marker, linewidths=0.8, label=label)
    _image_limits(ax, image)
    ax.axis("off")
    ax.set_title(title or f"{len(pos)} columns")
    if label:
        ax.legend(loc="upper right", fontsize=8)
    return fig, ax


def plot_cages(field, A, *, image=None, ax=None, title=None):
    """Show the complete four-corner cages and their centroids.

    Parameters
    ----------
    field : DisplacementField
        From :func:`polarmap.displacement.measure_cage_displacement`; its
        residual image is shown when ``image`` is not given.
    A : array_like
        The A-column positions used for the measurement.
    """
    fig, ax = get_axes(ax)
    background = field.diagnostics.get("residual") if image is None else image
    if background is not None:
        show_image(ax, background)
    A = np.asarray(A, dtype=float)
    order = [0, 1, 3, 2, 0]
    for corners in field.per_site.get("corner_indices", []):
        if np.all(corners >= 0):
            pts = A[corners][order]
            ax.plot(pts[:, 0], pts[:, 1], color="cyan", lw=0.4, alpha=0.6)
    ax.scatter(field.x, field.y, s=9, c="red", label="cage centroid (reference)")
    ax.scatter(*field.target_xy.T, s=12, facecolors="none", edgecolors="yellow",
               label="fitted target column")
    if background is not None:
        _image_limits(ax, background)
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title or f"{len(field)} complete cages")
    return fig, ax


def plot_pairs(field, reference_xy, *, image=None, ax=None, title=None):
    """Show complete reference pairs, ideal positions and fitted weak columns.

    Parameters
    ----------
    field : DisplacementField
        From :func:`polarmap.displacement.measure_pair_displacement`.
    reference_xy : array_like
        The bright reference columns.
    """
    fig, ax = get_axes(ax)
    background = field.diagnostics.get("residual") if image is None else image
    if background is not None:
        show_image(ax, background)
    ref = np.asarray(reference_xy, dtype=float)
    for i0, i1 in field.per_site.get("pair_indices", []):
        ax.plot(ref[[i0, i1], 0], ref[[i0, i1], 1], color="cyan", lw=0.5, alpha=0.6)
    ax.scatter(field.x, field.y, s=14, c="red", marker="+", linewidths=0.8,
               label="ideal position")
    ax.scatter(*field.target_xy.T, s=12, facecolors="none", edgecolors="yellow",
               linewidths=0.8, label="fitted weak column")
    if background is not None:
        _image_limits(ax, background)
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title or f"{len(field)} complete pairs")
    return fig, ax


def plot_intensity_split(image, split, *, ax=None, title="Fitted-amplitude split"):
    """Show the bright and dim groups from :func:`polarmap.columns.split_by_amplitude`."""
    fig, ax = get_axes(ax)
    show_image(ax, image)
    ax.scatter(*split.bright.T, s=18, facecolors="none", edgecolors="red",
               linewidths=0.8, label=f"bright ({len(split.bright)})")
    ax.scatter(*split.dim.T, s=12, facecolors="none", edgecolors="cyan",
               linewidths=0.7, label=f"dim ({len(split.dim)})")
    _image_limits(ax, image)
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title)
    return fig, ax


def plot_neighbor_diagnostic(graph, index, *, image=None, ax=None, zoom=None,
                             show_candidates=True):
    """Show which neighbour was matched in each direction for one column.

    Solid arrows are bonds that are used; dashed arrows were rejected by the
    loop-closure test. See also :meth:`polarmap.lattice.LatticeGraph.describe_column`.

    Parameters
    ----------
    graph : LatticeGraph
    index : int
        Column to inspect.
    image : array_like, optional
    ax : matplotlib.axes.Axes, optional
    zoom : float, optional
        Half-width of the view in pixels; default 2.5 lattice spacings.
    show_candidates : bool, default True
        Mark the other columns inside the search radius in grey.
    """
    fig, ax = get_axes(ax)
    if image is not None:
        show_image(ax, image)
    p = graph.positions[index]
    spacing = np.linalg.norm(graph.directions, axis=1).max()
    zoom = 2.5 * spacing if zoom is None else zoom
    if show_candidates:
        cand = KDTree(graph.positions).query_ball_point(p, r=2.2 * spacing)
        cand = [j for j in cand if j != index]
        if cand:
            ax.scatter(*graph.positions[cand].T, s=25, facecolors="none",
                       edgecolors="lightgray", label="candidates")
    ax.scatter(*p, s=90, c="lime", marker="*", zorder=5, label=f"column {index}")
    colors = ["tab:red", "tab:orange", "tab:blue", "tab:cyan"]
    for d, name in enumerate(DIRECTION_NAMES):
        j = graph.neighbor_idx[index, d]
        if j < 0:
            continue
        ok = not graph.bad_edges[index, d]
        end = p + graph.neighbor_vec[index, d]
        ax.annotate("", xy=tuple(end), xytext=tuple(p),
                    arrowprops=dict(arrowstyle="->", color=colors[d], lw=2,
                                    linestyle="-" if ok else "--"))
        ax.scatter(*end, s=40, c=colors[d], edgecolors="k", zorder=4,
                   label=f"{name}: {j} [{'used' if ok else 'rejected'}]")
    ax.set_xlim(p[0] - zoom, p[0] + zoom)
    ax.set_ylim(p[1] + zoom, p[1] - zoom)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_title(f"Neighbour matching for column {index}\n"
                 "(dashed = rejected by loop closure)")
    return fig, ax


def plot_reference_region(image, graph, reference, *, x_range=None, y_range=None,
                          ax=None, margin=None, title=None):
    """Show the reference region and which reference columns entered the fit.

    Parameters
    ----------
    image : array_like
    graph : LatticeGraph
    reference : polarmap.strain.ReferenceLattice
    x_range, y_range : (float, float), optional
        The reference rectangle (px) to outline.
    ax : matplotlib.axes.Axes, optional
    margin : float, optional
        Zoom to the reference region with this margin (px); default whole image.
    """
    fig, ax = get_axes(ax)
    show_image(ax, image)
    pos = graph.positions
    used = reference.used_mask
    rejected = reference.reference_mask & ~used
    ax.scatter(*pos[used].T, s=14, c="lime", label=f"used ({used.sum()})")
    ax.scatter(*pos[rejected].T, s=24, c="red", marker="x",
               label=f"rejected ({rejected.sum()})")
    ax.scatter(*pos[reference.anchor], s=90, c="yellow", marker="*",
               label="BFS anchor (0, 0)")
    if x_range is not None and y_range is not None:
        ax.add_patch(mpatches.Rectangle((min(x_range), min(y_range)),
                                        abs(x_range[1] - x_range[0]),
                                        abs(y_range[1] - y_range[0]),
                                        fill=False, ec="lime", lw=2))
    _image_limits(ax, image)
    if margin is not None and x_range is not None and y_range is not None:
        ax.set_xlim(min(x_range) - margin, max(x_range) + margin)
        ax.set_ylim(max(y_range) + margin, min(y_range) - margin)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title or "Reference lattice fit: columns used / rejected")
    return fig, ax


def plot_lattice_planes(image, positions, vector, *, n_planes=10, ax=None,
                        color="yellow", lw=1.0, title=None):
    """Draw evenly spaced lines perpendicular to a lattice vector.

    A quick visual check that ``vector`` is a genuine lattice direction: the
    lines should follow the atomic planes across the whole image.
    """
    fig, ax = get_axes(ax)
    show_image(ax, image)
    vec = np.asarray(vector, dtype=float)
    u = vec / np.linalg.norm(vec)
    t = np.array([-u[1], u[0]])
    proj = np.asarray(positions, dtype=float) @ u
    h, w = np.asarray(image).shape
    s = np.linspace(-1.5 * max(h, w), 1.5 * max(h, w), 2)
    for d in np.linspace(proj.min(), proj.max(), n_planes):
        ax.plot(d * u[0] + s * t[0], d * u[1] + s * t[1], color=color, lw=lw)
    _image_limits(ax, image)
    ax.axis("off")
    ax.set_title(title or f"planes perpendicular to ({vec[0]:.2f}, {vec[1]:.2f}) px")
    return fig, ax


def plot_neighbor_cloud(positions, *, n_neighbors=6, percentile_range=(20, 40),
                        ax=None, title="Neighbour displacement cloud"):
    """Scatter the vectors from each column to its nearest neighbours.

    The first-shell vectors form tight clusters at ``±v1`` and ``±v2``; broad
    or split clusters indicate distortion, mixed sublattices or false
    detections. By default only vectors between the 20th and 40th percentile of
    length are drawn (the first shell), as in the original notebook.
    """
    fig, ax = get_axes(ax, figsize=(5, 5))
    pos = np.asarray(positions, dtype=float)
    k = min(n_neighbors + 1, len(pos))
    _, idx = KDTree(pos).query(pos, k=k)
    vecs = (pos[idx[:, 1:]] - pos[:, None, :]).reshape(-1, 2)
    r = np.hypot(vecs[:, 0], vecs[:, 1])
    if percentile_range is not None:
        lo, hi = np.percentile(r, percentile_range)
        vecs = vecs[(r >= lo) & (r <= hi)]
    ax.scatter(vecs[:, 0], vecs[:, 1], s=5, alpha=0.3)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("dx (px)")
    ax.set_ylabel("dy (px)")
    ax.set_title(title)
    return fig, ax
