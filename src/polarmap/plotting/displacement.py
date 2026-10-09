"""Figures for displacement fields: arrow maps, colour maps and statistics."""

from __future__ import annotations

import numpy as np
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.interpolate import griddata
from matplotlib.axes import Axes
from matplotlib.text import Text

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


def _reserve_colorbar_space(ax, location="right"):
    """Reserve the same space as a colour bar without displaying one."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    empty_ax = divider.append_axes(location, size="4%", pad=0.08)
    empty_ax.set_axis_off()


def plot_displacement_vectors(field, image=None, *, color_by="angle",
                              sampling=None, unit="pm", ax=None, scale=10.0,
                              uniform_length=False, cmap=None, color="k",
                              halo="w", width=0.004, stroke=1.2,
                              convention="cartesian", colorbar_location="right",
                              scalebar_nm=None, dim_image=True, title=None,
                              clim=None):
    """Draw displacement vectors as arrows over the image.

    scale controls visual arrow magnification.
    uniform_length displays nonzero vectors with the median magnitude.
    Arrows are clipped at the axes boundary without changing field data.

    Returns
    -------
    fig, ax
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a positive finite number")

    fig, ax = get_axes(ax)

    if image is not None:
        show_image(ax, image, dim=dim_image)

    u, v = field.u.copy(), field.v.copy()

    if uniform_length:
        norm = np.hypot(u, v)
        norm[norm == 0] = 1.0
        length = np.median(np.hypot(field.u, field.v))
        u, v = u / norm * length, v / norm * length

    kw = dict(
        angles="xy",
        scale_units="xy",
        scale=1.0 / scale,
        width=width,
        headwidth=4,
        headlength=5,
        headaxislength=4,
        clip_on=True
    )

    if color_by == "angle":
        cmap = CYCLIC_CMAP if cmap is None else cmap

        q = ax.quiver(
            field.x, field.y, u, v, field.angle(convention),
            cmap=cmap, clim=(-180, 180), **kw
        )
        q.set_path_effects([
            pe.withStroke(linewidth=stroke, foreground="k")
        ])

        _finish_axes(ax, image, title, sampling, scalebar_nm)
        _reserve_colorbar_space(ax, colorbar_location)

        wheel = add_colorwheel(
            ax,
            cmap=cmap,
            convention=convention,
            bounds=(0.76, 0.77, 0.19, 0.19),
            label=""
        )
        wheel.tick_params(axis="x", labelsize=16, pad=2)

        for text in wheel.get_xticklabels():
            text.set_fontweight("bold")

    elif color_by == "magnitude":
        cmap = "plasma" if cmap is None else cmap
        mag, label = _magnitude(field, sampling, unit)

        if clim is None:
            clim = (
                0.0,
                float(np.nanpercentile(mag, 95)) if mag.size else 1.0
            )

        q = ax.quiver(
            field.x, field.y, u, v, mag,
            cmap=cmap, clim=clim, **kw
        )
        q.set_path_effects([
            pe.withStroke(linewidth=stroke, foreground="k")
        ])

        _finish_axes(ax, image, title, sampling, scalebar_nm)

        cbar = add_colorbar(
            q, ax, f"|displacement| ({label})",
            location=colorbar_location,
            size="4%",
            pad=0.08
        )
        cbar.set_label(
            f"|displacement| ({label})",
            fontsize=16,
            fontweight="bold",
            labelpad=10
        )
        cbar.ax.tick_params(axis="both", labelsize=16)

        for text in (
            cbar.ax.get_xticklabels()
            + cbar.ax.get_yticklabels()
        ):
            text.set_fontweight("bold")

        for axis in (cbar.ax.xaxis, cbar.ax.yaxis):
            axis.get_offset_text().set_fontsize(16)
            axis.get_offset_text().set_fontweight("bold")

    elif color_by is None:
        q = ax.quiver(
            field.x, field.y, u, v,
            color=color, **kw
        )
        q.set_path_effects([
            pe.withStroke(linewidth=stroke, foreground=halo)
        ])

        _finish_axes(ax, image, title, sampling, scalebar_nm)
        _reserve_colorbar_space(ax, colorbar_location)

    else:
        raise ValueError("color_by must be 'angle', 'magnitude' or None")

    ax.set_title(
        ax.get_title(),
        fontsize=16,
        fontweight="bold",
        pad=12
    )

    if image is not None:
        h, w = np.asarray(image).shape
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)

    ax.set_autoscale_on(False)
    q.set_clip_on(True)
    q.set_clip_path(ax.patch)
    q.set_clip_box(ax.bbox)

    return fig, ax


def _tile_polygons(xy, v1, v2):
    corners = np.array([
        -0.5 * v1 - 0.5 * v2,
         0.5 * v1 - 0.5 * v2,
         0.5 * v1 + 0.5 * v2,
        -0.5 * v1 + 0.5 * v2
    ])
    return xy[:, None, :] + corners[None, :, :]


def plot_displacement_map(field, quantity="magnitude", *, image=None,
                          sampling=None, unit="pm", kind="tiles",
                          lattice_vectors=None, ax=None, cmap=None, clim=None,
                          alpha=0.9, convention="cartesian",
                          colorbar_location="right", scalebar_nm=None,
                          title=None):
    """Colour map of one displacement quantity, without arrows.

    quantity selects magnitude, angle, u or v.
    kind selects tiles, interpolated or scatter rendering.
    sampling is the pixel size in nm/px.

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

        if clim is None:
            clim = (
                0.0,
                float(np.nanpercentile(values, 99))
                if len(values) else 1.0
            )

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
            vmax = (
                float(np.nanpercentile(np.abs(values), 99))
                if len(values) else 1.0
            )
            clim = (-vmax, vmax)

        direction = {
            "u": "x (right)",
            "v": "y (up)" if convention == "cartesian" else "y (down)"
        }
        label = f"displacement along {direction[quantity]} ({ulabel})"

    else:
        raise ValueError(
            "quantity must be 'magnitude', 'angle', 'u' or 'v'"
        )

    if kind == "tiles":
        if lattice_vectors is None:
            lattice_vectors = (
                estimate_lattice_vectors(xy)
                if len(xy) >= 4
                else (np.array([1.0, 0.0]), np.array([0.0, 1.0]))
            )

        v1, v2 = (
            np.asarray(v, dtype=float)
            for v in lattice_vectors
        )

        mappable = PolyCollection(
            _tile_polygons(xy, v1, v2),
            array=values,
            cmap=cmap,
            alpha=alpha,
            edgecolors="face",
            linewidths=0.3
        )
        mappable.set_clim(*clim)
        ax.add_collection(mappable, autolim=True)

    elif kind == "interpolated":
        if image is not None:
            h, w = np.asarray(image).shape
            gx, gy = np.meshgrid(np.arange(w), np.arange(h))
        else:
            gx, gy = np.meshgrid(
                np.arange(
                    np.floor(xy[:, 0].min()),
                    np.ceil(xy[:, 0].max()) + 1
                ),
                np.arange(
                    np.floor(xy[:, 1].min()),
                    np.ceil(xy[:, 1].max()) + 1
                )
            )

        if quantity == "angle":
            # Interpolate unit vectors to respect angular wrap-around.
            rad = np.deg2rad(values)
            c = griddata(
                xy, np.cos(rad), (gx, gy), method="linear"
            )
            s = griddata(
                xy, np.sin(rad), (gx, gy), method="linear"
            )
            grid = np.degrees(np.arctan2(s, c))
        else:
            grid = griddata(
                xy, values, (gx, gy), method="linear"
            )

        mappable = ax.imshow(
            grid,
            cmap=cmap,
            vmin=clim[0],
            vmax=clim[1],
            alpha=alpha,
            interpolation="nearest",
            extent=(
                gx.min() - 0.5,
                gx.max() + 0.5,
                gy.max() + 0.5,
                gy.min() - 0.5
            )
        )

    elif kind == "scatter":
        mappable = ax.scatter(
            xy[:, 0], xy[:, 1],
            c=values,
            cmap=cmap,
            marker="s",
            s=40,
            vmin=clim[0],
            vmax=clim[1],
            alpha=alpha,
            edgecolors="none"
        )

    else:
        raise ValueError(
            "kind must be 'tiles', 'interpolated' or 'scatter'"
        )

    _finish_axes(ax, image, title, sampling, scalebar_nm)

    ax.set_title(
        ax.get_title(),
        fontsize=16,
        fontweight="bold",
        pad=12
    )

    if quantity == "angle":
        _reserve_colorbar_space(ax, colorbar_location)

        wheel = add_colorwheel(
            ax,
            cmap=cmap,
            convention=convention,
            bounds=(0.76, 0.77, 0.19, 0.19),
            label=""
        )
        wheel.tick_params(axis="x", labelsize=16, pad=2)

        for text in wheel.get_xticklabels():
            text.set_fontweight("bold")

    else:
        cbar = add_colorbar(
            mappable, ax, label,
            location=colorbar_location,
            size="4%",
            pad=0.08
        )
        cbar.set_label(
            label,
            fontsize=16,
            fontweight="bold",
            labelpad=10
        )
        cbar.ax.tick_params(axis="both", labelsize=16)

        for text in (
            cbar.ax.get_xticklabels()
            + cbar.ax.get_yticklabels()
        ):
            text.set_fontweight("bold")

        for axis in (cbar.ax.xaxis, cbar.ax.yaxis):
            axis.get_offset_text().set_fontsize(16)
            axis.get_offset_text().set_fontweight("bold")

    return fig, ax

    


def plot_displacement_statistics(field, sampling, *, n_mad=6.0,
                                 convention="cartesian", bins=30, axes=None,
                                 color_magnitude="tab:blue",
                                 color_angle="tab:green"):
    """Show two stacked histograms and a larger orientation rose.

    Existing axes retain their layout. The last axis must be polar.

    Returns
    -------
    fig, axes, descriptors
    """
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    desc = describe_displacements(
        field, sampling,
        n_mad=n_mad,
        convention=convention
    )
    mag = field.magnitude_in(sampling, "pm")[desc["kept"]]
    ang = field.angle(convention)[desc["kept"]]

    if axes is None:
        fig = plt.figure(figsize=(18, 10), constrained_layout=True)
        gs = fig.add_gridspec(
            2, 2,
            width_ratios=(1.0, 1.25)
        )

        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[:, 1], projection="polar")
        ]
    else:
        fig = axes[0].figure

    ax0, ax1, ax2 = axes

    # Magnitude histogram
    ax0.hist(
        mag, bins=bins,
        color=color_magnitude,
        edgecolor="k", alpha=0.8
    )
    ax0.axvline(
        desc["median_pm"],
        color="r", ls="--",
        label="median"
    )
    ax0.set_xlabel("displacement |d| (pm)")
    ax0.set_ylabel("count")
    ax0.set_title("magnitude distribution", pad=12)
    ax0.legend()

    # Fewer labels without changing histogram bins.
    ax0.xaxis.set_major_locator(MaxNLocator(nbins=4))
    formatter = ScalarFormatter(useOffset=False)
    ax0.xaxis.set_major_formatter(formatter)
    ax0.yaxis.set_major_locator(
        MaxNLocator(nbins=5, integer=True)
    )

    # Orientation histogram
    ax1.hist(
        np.mod(ang, 360),
        bins=np.arange(0, 361, 10),
        color=color_angle,
        edgecolor="k", alpha=0.8
    )
    ax1.set_xlabel(f"orientation ({convention}, deg)")
    ax1.set_ylabel("count")
    ax1.set_xticks([0, 90, 180, 270, 360])
    ax1.set_xlim(-5, 365)
    ax1.set_title("orientation distribution", pad=12)
    ax1.yaxis.set_major_locator(
        MaxNLocator(nbins=5, integer=True)
    )

    # Orientation rose: radius represents the number of vectors.
    counts, edges = np.histogram(
        np.mod(np.deg2rad(ang), 2 * np.pi),
        bins=36,
        range=(0, 2 * np.pi)
    )

    ax2.bar(
        0.5 * (edges[:-1] + edges[1:]),
        counts,
        width=np.diff(edges),
        color=color_angle,
        edgecolor="k",
        alpha=0.7
    )

    ax2.set_theta_zero_location("E")
    ax2.set_theta_direction(
        1 if convention == "cartesian" else -1
    )
    ax2.set_title("orientation rose — count", pad=24)

    radial_max = max(1.0, float(counts.max()) * 1.12)
    ax2.set_ylim(0, radial_max)

    radial_locator = MaxNLocator(nbins=4, integer=True)
    radial_ticks = radial_locator.tick_values(0, radial_max)
    radial_ticks = radial_ticks[
        (radial_ticks > 0) & (radial_ticks < radial_max)
    ]

    ax2.set_yticks(radial_ticks)
    ax2.set_rlabel_position(22.5)
    ax2.tick_params(axis="x", pad=10)
    ax2.grid(alpha=0.5)

    title_size = 20
    label_size = 18
    tick_size = 16
    legend_size = 16

    for ax in axes:
        ax.title.set_fontsize(title_size)
        ax.title.set_fontweight("bold")

        for label in (ax.xaxis.label, ax.yaxis.label):
            label.set_fontsize(label_size)
            label.set_fontweight("bold")

        ax.tick_params(
            axis="both", which="both",
            labelsize=tick_size
        )

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

        for axis in (ax.xaxis, ax.yaxis):
            axis.get_offset_text().set_fontsize(tick_size)
            axis.get_offset_text().set_fontweight("bold")

        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(legend_size)
                text.set_fontweight("bold")

            legend.get_title().set_fontsize(legend_size)
            legend.get_title().set_fontweight("bold")

    return fig, axes, desc

def plot_displacement_polar(field, sampling, *, n_mad=6.0,
                            convention="cartesian", ax=None,
                            color="tab:green", s=20, alpha=0.5):
    """Plot displacement direction against magnitude in pm.

    Each point represents one retained displacement vector.
    Spatial positions are not represented.

    Returns
    -------
    fig, ax, descriptors
    """
    from matplotlib.ticker import MaxNLocator

    desc = describe_displacements(
        field, sampling,
        n_mad=n_mad,
        convention=convention
    )

    mag = field.magnitude_in(sampling, "pm")[desc["kept"]]
    ang = field.angle(convention)[desc["kept"]]

    valid = np.isfinite(mag) & np.isfinite(ang)
    radii = mag[valid]
    theta = np.mod(np.deg2rad(ang[valid]), 2 * np.pi)

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(9, 9),
            dpi=120,
            subplot_kw={"projection": "polar"},
            constrained_layout=True
        )
    else:
        if getattr(ax, "name", None) != "polar":
            raise ValueError("ax must use a polar projection")
        fig = ax.figure

    ax.scatter(
        theta, radii,
        s=s,
        color=color,
        alpha=alpha,
        edgecolors="none"
    )

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(
        1 if convention == "cartesian" else -1
    )

    max_mag = float(radii.max()) if radii.size else 0.0
    radial_max = 1.05 * max_mag if max_mag > 0 else 1.0
    ax.set_ylim(0, radial_max)

    locator = MaxNLocator(nbins=4)
    ticks = locator.tick_values(0, radial_max)
    ticks = ticks[(ticks > 0) & (ticks < radial_max)]

    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{value:g} pm" for value in ticks])
    ax.set_rlabel_position(22.5)

    ax.set_title(
        "displacement magnitude and direction",
        fontsize=20,
        fontweight="bold",
        pad=24
    )
    ax.tick_params(axis="both", labelsize=16)
    ax.tick_params(axis="x", pad=10)

    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontweight("bold")

    ax.grid(alpha=0.5)

    return fig, ax, desc

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
