"""Diagnostic figures: check each step before interpreting a map."""

from __future__ import annotations

import numpy as np
import matplotlib.patches as mpatches
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

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


def plot_columns(image, positions, *, ax=None, color="lime", s=12,
                 marker="o", title=None, label=None,
                 crop_size=256, zoom_s=70):
    """Display column positions on the full image and a central detail.

    Coordinates must be supplied as (x, y), in pixels.
    This function only displays results; it does not modify them.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    image = np.asarray(image)
    pos = np.asarray(positions, dtype=float).reshape(-1, 2)

    # Without an existing axis, create two panels.
    if ax is None:
        fig, (ax, ax_zoom) = plt.subplots(
            1, 2, figsize=(14, 7), dpi=120,
            constrained_layout=True
        )
    else:
        fig = ax.figure
        ax_zoom = ax.inset_axes([0.62, 0.03, 0.35, 0.35])

    # Central region, expressed in the original image coordinates.
    h, w = image.shape
    size = max(1, int(crop_size))
    ch, cw = min(size, h), min(size, w)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    y1, x1 = y0 + ch, x0 + cw

    # Full image: resampling for display only.
    show_image(ax, image, interpolation="hanning")
    ax.scatter(
        pos[:, 0], pos[:, 1],
        s=s, facecolors="none", edgecolors=color,
        marker=marker, linewidths=0.8, label=label
    )

    ax.add_patch(Rectangle(
        (x0 - 0.5, y0 - 0.5), cw, ch,
        fill=False, edgecolor="cyan", linewidth=1.2
    ))

    _image_limits(ax, image)
    ax.axis("off")
    ax.set_title(title or f"{len(pos)} columns")

    if label:
        ax.legend(loc="upper right", fontsize=8)

    # Select positions located inside the central region.
    inside = (
        (pos[:, 0] >= x0 - 0.5) &
        (pos[:, 0] < x1 - 0.5) &
        (pos[:, 1] >= y0 - 0.5) &
        (pos[:, 1] < y1 - 0.5)
    )
    selected = pos[inside]

    # Display the same image and contrast, zoomed into the central region.
    show_image(ax_zoom, image, interpolation="nearest")
    ax_zoom.scatter(
        selected[:, 0], selected[:, 1],
        s=zoom_s, facecolors="none", edgecolors=color,
        marker=marker, linewidths=1.0
    )

    ax_zoom.set_xlim(x0 - 0.5, x1 - 0.5)
    ax_zoom.set_ylim(y1 - 0.5, y0 - 0.5)
    ax_zoom.set_title(
        f"Central detail — {cw} × {ch} px\n"
        f"{len(selected)} columns"
    )
    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])

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


def plot_reference_region(image, graph, reference, *,
                          x_range=None, y_range=None,
                          ax=None, margin=None, title=None):
    """Show the full image and a separate reference-region detail."""
    from matplotlib.patches import Rectangle

    image = np.asarray(image)
    pos = np.asarray(graph.positions)
    used = np.asarray(reference.used_mask, dtype=bool)
    ref_mask = np.asarray(reference.reference_mask, dtype=bool)
    rejected = ref_mask & ~used

    # Determine the reference-region bounds.
    ref_pos = pos[ref_mask]
    if len(ref_pos) == 0:
        raise ValueError("The reference region contains no columns.")

    xmin, xmax = (
        sorted(x_range) if x_range is not None
        else (ref_pos[:, 0].min(), ref_pos[:, 0].max())
    )
    ymin, ymax = (
        sorted(y_range) if y_range is not None
        else (ref_pos[:, 1].min(), ref_pos[:, 1].max())
    )

    pad = 20.0 if margin is None else float(margin)
    if pad < 0:
        raise ValueError("margin must be non-negative.")

    # Respect an existing axis; otherwise create two panels.
    if ax is None:
        fig, (ax, ax_ref) = plt.subplots(
            1, 2, figsize=(14, 7), dpi=120,
            constrained_layout=True
        )
    else:
        fig = ax.figure
        detail_fig, ax_ref = plt.subplots(
            figsize=(7, 7), dpi=120,
            constrained_layout=True
        )

    # Full image with the reference rectangle.
    show_image(ax, image, interpolation="hanning")
    ax.add_patch(Rectangle(
        (xmin, ymin), xmax - xmin, ymax - ymin,
        fill=False, edgecolor="lime", linewidth=1.5
    ))
    _image_limits(ax, image)
    ax.set_title("Full image — reference region")
    ax.axis("off")

    # Same image and contrast, zoomed into the reference region.
    show_image(ax_ref, image, interpolation="nearest")

    ax_ref.scatter(
        *pos[used].T,
        s=35, facecolors="none", edgecolors="lime",
        linewidths=1.0, label=f"used ({used.sum()})"
    )
    ax_ref.scatter(
        *pos[rejected].T,
        s=40, c="red", marker="x",
        linewidths=1.2, label=f"rejected ({rejected.sum()})"
    )
    ax_ref.scatter(
        *pos[reference.anchor],
        s=110, c="yellow", marker="*",
        edgecolors="black", linewidths=0.6,
        label="BFS anchor (0, 0)"
    )

    h, w = image.shape
    ax_ref.set_xlim(
        max(-0.5, xmin - pad),
        min(w - 0.5, xmax + pad)
    )
    ax_ref.set_ylim(
        min(h - 0.5, ymax + pad),
        max(-0.5, ymin - pad)
    )

    ax_ref.legend(loc="upper right", fontsize=8)
    ax_ref.set_title(title or "Reference region — used / rejected")
    ax_ref.axis("off")

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
