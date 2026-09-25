"""Figures for displacement fields: arrow maps, colour maps and statistics."""

from __future__ import annotations

import numpy as np
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.interpolate import griddata

from ..lattice import estimate_lattice_vectors
from ..statistics import describe_displacements
from ._common import (CYCLIC_CMAP, add_colorbar, add_colorwheel, add_scalebar,
                      get_axes, show_image)

__all__ = [
    "plot_displacement_vectors",
    "plot_displacement_map",
    "plot_displacement_statistics",
    "plot_magnitude_agreement",
]

_UNIT_FACTORS = {"pm": 1e3, "nm": 1.0}


def _magnitude(field, sampling, unit):
    if sampling is None:
        return field.magnitude, "px"
    return field.magnitude * sampling * _UNIT_FACTORS[unit], unit


def _finish_axes(ax, image, title, sampling, scalebar_nm):
    if image is not None:
        h, w = np.asarray(image).shape
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
    else:
        ax.autoscale_view()
        if not ax.yaxis_inverted():          # image convention: y down
            ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)
    if scalebar_nm is not None:
        if sampling is None:
            raise ValueError("a scale bar needs the sampling (nm/px)")
        add_scalebar(ax, sampling, None if scalebar_nm == "auto" else scalebar_nm,
                     color="black" if image is None else "white")


def plot_displacement_vectors(field, image=None, *, color_by="angle",
                              sampling=None, unit="pm", ax=None, scale=10.0,
                              uniform_length=False, cmap=None, color="k",
                              halo="w", width=0.004, stroke=1.2,
                              convention="cartesian", colorbar_location="right",
                              scalebar_nm=None, dim_image=True, title=None,
                              clim=None):
    """Draw displacement vectors as arrows over the image.

    Parameters
    ----------
    field : DisplacementField
    image : array_like, optional
        Background image (shown dimmed by default so the arrows stay visible).
    color_by : {"angle", "magnitude", None}, default "angle"
        ``"angle"``: cyclic colour map with a colour-wheel key; ``"magnitude"``:
        sequential colour map with a colour bar; ``None``: single colour
        ``color`` with a contrasting ``halo``.
    sampling : float, optional
        Pixel size in nm/px; needed for magnitudes in pm/nm and for scale bars.
    unit : {"pm", "nm"}, default "pm"
    ax : matplotlib.axes.Axes, optional
    scale : float, default 10
        Arrow length magnification (displacements are much smaller than the
        lattice spacing).
    uniform_length : bool, default False
        Draw all arrows with the median length (direction only), which is more
        readable when magnitudes vary strongly.
    cmap : str, optional
        Colour map; default cyclic for angles, ``"plasma"`` for magnitudes.
    color, halo : str
        Arrow colour and outline colour for ``color_by=None``.
    width : float, default 0.004
        Arrow shaft width (fraction of the axes width).
    stroke : float, default 1.2
        Outline width; outlines keep arrows legible over bright and dark
        columns.
    convention : {"cartesian", "image"}, default "cartesian"
        Angle convention for colour coding (see :mod:`polarmap.statistics`).
    colorbar_location : str, default "right"
        See :func:`~polarmap.plotting.add_colorbar`.
    scalebar_nm : float or "auto", optional
        Add a scale bar of this length.
    dim_image : bool, default True
    title : str, optional
    clim : (float, float), optional
        Colour limits for magnitudes (default: 0 to the 95th percentile).

    Returns
    -------
    fig, ax
    """
    fig, ax = get_axes(ax)
    if image is not None:
        show_image(ax, image, dim=dim_image)
    u, v = field.u.copy(), field.v.copy()
    if uniform_length:
        norm = np.hypot(u, v)
        norm[norm == 0] = 1.0
        length = np.median(np.hypot(field.u, field.v))
        u, v = u / norm * length, v / norm * length
    kw = dict(angles="xy", scale_units="xy", scale=1.0 / scale, width=width,
              headwidth=4, headlength=5, headaxislength=4)

    if color_by == "angle":
        cmap = CYCLIC_CMAP if cmap is None else cmap
        q = ax.quiver(field.x, field.y, u, v, field.angle(convention), cmap=cmap,
                      clim=(-180, 180), **kw)
        q.set_path_effects([pe.withStroke(linewidth=stroke, foreground="k")])
        _finish_axes(ax, image, title, sampling, scalebar_nm)
        add_colorwheel(ax, cmap=cmap, convention=convention,
                       label=f"direction ({convention})")
    elif color_by == "magnitude":
        cmap = "plasma" if cmap is None else cmap
        mag, label = _magnitude(field, sampling, unit)
        if clim is None:
            clim = (0.0, float(np.nanpercentile(mag, 95)) if mag.size else 1.0)
        q = ax.quiver(field.x, field.y, u, v, mag, cmap=cmap, clim=clim, **kw)
        q.set_path_effects([pe.withStroke(linewidth=stroke, foreground="k")])
        _finish_axes(ax, image, title, sampling, scalebar_nm)
        add_colorbar(q, ax, f"|displacement| ({label})",
                     location=colorbar_location)
    elif color_by is None:
        q = ax.quiver(field.x, field.y, u, v, color=color, **kw)
        q.set_path_effects([pe.withStroke(linewidth=stroke, foreground=halo)])
        _finish_axes(ax, image, title, sampling, scalebar_nm)
    else:
        raise ValueError("color_by must be 'angle', 'magnitude' or None")
    return fig, ax


def _tile_polygons(xy, v1, v2):
    corners = np.array([-0.5 * v1 - 0.5 * v2, 0.5 * v1 - 0.5 * v2,
                        0.5 * v1 + 0.5 * v2, -0.5 * v1 + 0.5 * v2])
    return xy[:, None, :] + corners[None, :, :]


def plot_displacement_map(field, quantity="magnitude", *, image=None,
                          sampling=None, unit="pm", kind="tiles",
                          lattice_vectors=None, ax=None, cmap=None, clim=None,
                          alpha=0.9, convention="cartesian",
                          colorbar_location="right", scalebar_nm=None,
                          title=None):
    """Colour map of one displacement quantity, without arrows.

    When only the magnitude or only the orientation is of interest, a colour
    map is easier to read than arrows: each unit cell is filled with a colour
    encoding the value.

    Parameters
    ----------
    field : DisplacementField
    quantity : {"magnitude", "angle", "u", "v"}, default "magnitude"
        ``"u"``/``"v"`` are the signed components (``v`` positive **down** in
        image axes; for ``convention="cartesian"``, ``-v`` is shown so that
        positive means up).
    image : array_like, optional
        Background image (shown underneath; set ``alpha < 1`` to see it).
    sampling : float, optional
        nm/px; needed for physical units and scale bars.
    unit : {"pm", "nm"}, default "pm"
    kind : {"tiles", "interpolated", "scatter"}, default "tiles"
        ``"tiles"``: one parallelogram per unit cell (spanned by the lattice
        vectors, centred on the reference position); ``"interpolated"``:
        linear interpolation between cells (convex hull only); ``"scatter"``:
        one square marker per cell.
    lattice_vectors : (v1, v2), optional
        Tile shape; estimated from the reference positions if omitted.
    ax : matplotlib.axes.Axes, optional
    cmap : str, optional
        Default: cyclic for ``"angle"``, ``"viridis"`` for ``"magnitude"``,
        ``"RdBu_r"`` for components.
    clim : (float, float), optional
    alpha : float, default 0.9
    convention : {"cartesian", "image"}, default "cartesian"
    colorbar_location : str, default "right"
    scalebar_nm : float or "auto", optional
    title : str, optional

    Returns
    -------
    fig, ax
    """
    fig, ax = get_axes(ax)
    if image is not None:
        show_image(ax, image)
    xy = field.reference_xy
    k = 1.0 if sampling is None else sampling * _UNIT_FACTORS[unit]
    ulabel = "px" if sampling is None else unit
    if quantity == "magnitude":
        values = field.magnitude * k
        cmap = cmap or "viridis"
        clim = clim or (0.0, float(np.nanpercentile(values, 99)) if len(values) else 1.0)
        label = f"|displacement| ({ulabel})"
    elif quantity == "angle":
        values = field.angle(convention)
        cmap = cmap or CYCLIC_CMAP
        clim = (-180.0, 180.0)
        label = None
    elif quantity in ("u", "v"):
        values = getattr(field, quantity) * k
        if quantity == "v" and convention == "cartesian":
            values = -values
        cmap = cmap or "RdBu_r"
        if clim is None:
            vmax = float(np.nanpercentile(np.abs(values), 99)) if len(values) else 1.0
            clim = (-vmax, vmax)
        direction = {"u": "x (right)",
                     "v": "y (up)" if convention == "cartesian" else "y (down)"}
        label = f"displacement along {direction[quantity]} ({ulabel})"
    else:
        raise ValueError("quantity must be 'magnitude', 'angle', 'u' or 'v'")

    if kind == "tiles":
        if lattice_vectors is None:
            lattice_vectors = estimate_lattice_vectors(xy) if len(xy) >= 4 else (
                np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        v1, v2 = (np.asarray(v, dtype=float) for v in lattice_vectors)
        mappable = PolyCollection(_tile_polygons(xy, v1, v2), array=values,
                                  cmap=cmap, alpha=alpha, edgecolors="face",
                                  linewidths=0.3)
        mappable.set_clim(*clim)
        ax.add_collection(mappable, autolim=True)
    elif kind == "interpolated":
        if image is not None:
            h, w = np.asarray(image).shape
            gx, gy = np.meshgrid(np.arange(w), np.arange(h))
        else:
            gx, gy = np.meshgrid(np.arange(np.floor(xy[:, 0].min()),
                                           np.ceil(xy[:, 0].max()) + 1),
                                 np.arange(np.floor(xy[:, 1].min()),
                                           np.ceil(xy[:, 1].max()) + 1))
        if quantity == "angle":
            # interpolate unit vectors, not angles, to respect the wrap-around
            rad = np.deg2rad(values)
            c = griddata(xy, np.cos(rad), (gx, gy), method="linear")
            s = griddata(xy, np.sin(rad), (gx, gy), method="linear")
            grid = np.degrees(np.arctan2(s, c))
        else:
            grid = griddata(xy, values, (gx, gy), method="linear")
        mappable = ax.imshow(grid, cmap=cmap, vmin=clim[0], vmax=clim[1],
                             alpha=alpha, interpolation="nearest",
                             extent=(gx.min() - 0.5, gx.max() + 0.5,
                                     gy.max() + 0.5, gy.min() - 0.5))
    elif kind == "scatter":
        mappable = ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap=cmap, marker="s",
                              s=40, vmin=clim[0], vmax=clim[1], alpha=alpha,
                              edgecolors="none")
    else:
        raise ValueError("kind must be 'tiles', 'interpolated' or 'scatter'")

    _finish_axes(ax, image, title, sampling, scalebar_nm)
    if quantity == "angle":
        add_colorwheel(ax, cmap=cmap, convention=convention,
                       label=f"direction ({convention})")
    else:
        add_colorbar(mappable, ax, label, location=colorbar_location)
    return fig, ax


def plot_displacement_statistics(field, sampling, *, n_mad=6.0,
                                 convention="cartesian", bins=30, axes=None,
                                 color_magnitude="tab:blue",
                                 color_angle="tab:green"):
    """Magnitude histogram, orientation histogram and orientation rose.

    The same outlier rule as :func:`polarmap.statistics.describe_displacements`
    is applied first.

    Parameters
    ----------
    field : DisplacementField
    sampling : float
        nm/px.
    n_mad : float, default 6
    convention : {"cartesian", "image"}, default "cartesian"
    bins : int, default 30
        Magnitude bins (orientations use 10-degree bins).
    axes : sequence of 3 Axes, optional
        The last one must use a polar projection.

    Returns
    -------
    fig, axes, descriptors
        ``descriptors`` is the dict from ``describe_displacements``.
    """
    desc = describe_displacements(field, sampling, n_mad=n_mad,
                                  convention=convention)
    mag = field.magnitude_in(sampling, "pm")[desc["kept"]]
    ang = field.angle(convention)[desc["kept"]]
    if axes is None:
        fig = plt.figure(figsize=(15, 4.2))
        axes = [fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2),
                fig.add_subplot(1, 3, 3, projection="polar")]
    else:
        fig = axes[0].figure
    ax0, ax1, ax2 = axes
    ax0.hist(mag, bins=bins, color=color_magnitude, edgecolor="k", alpha=0.8)
    ax0.axvline(desc["median_pm"], color="r", ls="--", label="median")
    ax0.set_xlabel("displacement |d| (pm)")
    ax0.set_ylabel("count")
    ax0.set_title("magnitude distribution")
    ax0.legend()

    ax1.hist(np.mod(ang, 360), bins=np.arange(0, 361, 10), color=color_angle,
             edgecolor="k", alpha=0.8)
    ax1.set_xlabel(f"orientation ({convention}, deg)")
    ax1.set_ylabel("count")
    ax1.set_xticks([0, 90, 180, 270, 360])
    ax1.set_title("orientation distribution")

    counts, edges = np.histogram(np.mod(np.deg2rad(ang), 2 * np.pi), bins=36,
                                 range=(0, 2 * np.pi))
    ax2.bar(0.5 * (edges[:-1] + edges[1:]), counts, width=2 * np.pi / 36,
            color=color_angle, edgecolor="k", alpha=0.7)
    ax2.set_theta_zero_location("E")
    ax2.set_theta_direction(1 if convention == "cartesian" else -1)
    ax2.set_title("orientation rose", pad=15)
    fig.tight_layout()
    return fig, axes, desc


def plot_magnitude_agreement(comparison, *, ax=None, labels=("polarmap", "VecMap"),
                             title=None):
    """Scatter plot of matched magnitudes from two analyses.

    Parameters
    ----------
    comparison : dict
        Output of :func:`polarmap.validation.compare_displacement_fields`.
    ax : matplotlib.axes.Axes, optional
    labels : (str, str)
        Names of the two analyses (first = ``field``, second = ``other``).
    title : str, optional

    Returns
    -------
    fig, ax
    """
    fig, ax = get_axes(ax, figsize=(5.5, 5.5))
    ours = comparison["magnitude_pm"]
    other = comparison["magnitude_other_pm"]
    lo = min(np.min(ours), np.min(other))
    hi = max(np.max(ours), np.max(other))
    pad = 0.05 * (hi - lo) if hi > lo else 0.05
    ax.scatter(other, ours, s=30, color="tab:blue", edgecolor="k", linewidth=0.4,
               alpha=0.85, label="matched columns")
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="tab:orange",
            label="perfect agreement")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{labels[1]} |d| (pm)")
    ax.set_ylabel(f"{labels[0]} |d| (pm)")
    ax.legend(frameon=False)
    ax.set_title(title or f"n = {comparison['n_matched']}, bias "
                          f"{comparison['magnitude_bias_pm']:+.2f} pm, RMSE "
                          f"{comparison['magnitude_rmse_pm']:.2f} pm")
    return fig, ax
