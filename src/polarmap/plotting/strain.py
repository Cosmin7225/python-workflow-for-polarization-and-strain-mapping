"""Figures for strain: atom-resolved maps, tensor panels, profiles."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from ..strain import interpolate_to_grid
from ._common import (add_colorbar, add_scalebar, data_aspect, get_axes,
                      show_image, symmetric_limits)

__all__ = [
    "plot_strain_scatter",
    "plot_strain_triangulation",
    "plot_strain_tensor",
    "plot_strain_comparison",
    "plot_profiles",
]

_TITLES = {"exx": r"$\varepsilon_{xx}$", "eyy": r"$\varepsilon_{yy}$",
           "exy": r"$\varepsilon_{xy}$", "omega": r"$\omega_{xy}$"}


def _coords(positions, sampling):
    pos = np.asarray(positions, dtype=float)
    return (pos, "px") if sampling is None else (pos * sampling, "nm")


def _style_map_axes(ax, unit, show_axes, title):
    ax.set_aspect("equal")
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    if show_axes:
        ax.set_xlabel(f"x ({unit})")
        ax.set_ylabel(f"y ({unit})")
    else:
        ax.axis("off")
    if title:
        ax.set_title(title)


def plot_strain_scatter(positions, values, *, image=None, sampling=None, ax=None,
                        percent=True, cmap="RdBu_r", clim=None, s=8,
                        label="strain", colorbar_location="right",
                        show_axes=True, scalebar_nm=None, title=None):
    """Atom-resolved strain map: one coloured marker per column.

    This is the most direct view of the measurement (no interpolation);
    compare smoothed maps against it before interpreting a feature.

    Parameters
    ----------
    positions : array_like
        ``(N, 2)`` column positions (px).
    values : array_like
        ``(N,)`` strain (dimensionless) or any scalar; NaNs are not drawn.
    image : array_like, optional
        Background image (then axes stay in pixels).
    sampling : float, optional
        nm/px: axes in nm (without ``image``) and scale bar support.
    ax : matplotlib.axes.Axes, optional
    percent : bool, default True
        Show values multiplied by 100.
    cmap : str, default "RdBu_r"
        Diverging map: red = tensile, blue = compressive.
    clim : (float, float), optional
        Colour limits (in displayed units); default symmetric at the 98th
        percentile of the absolute values.
    s : float, default 8
        Marker size.
    label : str, default "strain"
    colorbar_location : str, default "right"
    show_axes : bool, default True
    scalebar_nm : float or "auto", optional
    title : str, optional

    Returns
    -------
    fig, ax
    """
    fig, ax = get_axes(ax, figsize=(7, 6))
    vals = np.asarray(values, dtype=float) * (100.0 if percent else 1.0)
    if image is not None:
        show_image(ax, image)
        xy, unit = np.asarray(positions, dtype=float), "px"
    else:
        xy, unit = _coords(positions, sampling)
    ok = np.isfinite(vals)
    clim = clim or symmetric_limits(vals[ok])
    sc = ax.scatter(xy[ok, 0], xy[ok, 1], c=vals[ok], s=s, cmap=cmap,
                    vmin=clim[0], vmax=clim[1], edgecolors="none")
    _style_map_axes(ax, unit, show_axes and image is None, title)
    if image is not None:
        h, w = np.asarray(image).shape
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
    add_colorbar(sc, ax, f"{label} (%)" if percent else label,
                 location=colorbar_location)
    if scalebar_nm is not None:
        add_scalebar(ax, sampling, None if scalebar_nm == "auto" else scalebar_nm,
                     data_units="px" if unit == "px" else "nm",
                     color="white" if image is not None else "black")
    return fig, ax


def plot_strain_triangulation(positions, values, *, image=None, sampling=None,
                              ax=None, percent=True, cmap="RdBu_r", clim=None,
                              alpha=1.0, shading="gouraud", label="strain",
                              colorbar_location="right", show_axes=True,
                              scalebar_nm=None, title=None):
    """Continuous strain map by linear interpolation on the Delaunay triangulation.

    Useful for presentation; more sensitive to single outliers than the
    atom-resolved scatter. Parameters as in :func:`plot_strain_scatter`;
    ``shading`` is ``"gouraud"`` (smooth) or ``"flat"``.

    Returns
    -------
    fig, ax
    """
    fig, ax = get_axes(ax, figsize=(7, 6))
    vals = np.asarray(values, dtype=float) * (100.0 if percent else 1.0)
    if image is not None:
        show_image(ax, image)
        xy, unit = np.asarray(positions, dtype=float), "px"
    else:
        xy, unit = _coords(positions, sampling)
    ok = np.isfinite(vals)
    if ok.sum() < 3:
        raise ValueError("need at least 3 finite values")
    clim = clim or symmetric_limits(vals[ok])
    tri = mtri.Triangulation(xy[ok, 0], xy[ok, 1])
    tc = ax.tripcolor(tri, vals[ok], shading=shading, cmap=cmap, vmin=clim[0],
                      vmax=clim[1], alpha=alpha)
    _style_map_axes(ax, unit, show_axes and image is None, title)
    add_colorbar(tc, ax, f"{label} (%)" if percent else label,
                 location=colorbar_location)
    if scalebar_nm is not None:
        add_scalebar(ax, sampling, None if scalebar_nm == "auto" else scalebar_nm,
                     data_units="px" if unit == "px" else "nm",
                     color="white" if image is not None else "black")
    return fig, ax


def plot_strain_tensor(tensor, *, kind="scatter", sampling=None,
                       components=("exx", "eyy", "exy", "omega"), percent=True,
                       omega_unit="deg", step=2.0, cmap="RdBu_r",
                       clim_percentile=98.0, clims=None,
                       colorbar_location="bottom", axes=None, figsize=None,
                       s=6, show_axes=True, scalebar_nm=None, titles=None,
                       title_size=20, label_size=18, tick_size=16):
    """Multi-panel figure of strain-tensor components (Method 2).

    Font sizes apply to titles, axis labels and colour-bar labels/ticks.
    All text is displayed in bold.

    Returns
    -------
    fig, axes
    """
    from matplotlib.text import Text

    components = list(components)

    if axes is None:
        if figsize is None:
            panel = 4.4
            extra = 1.5 if show_axes else 1.0
            figsize = (
                panel * len(components),
                panel * min(data_aspect(tensor.positions), 1.6) + extra
            )

        fig, axes = plt.subplots(
            1, len(components), figsize=figsize
        )
        axes = np.atleast_1d(axes)
    else:
        axes = np.atleast_1d(axes).ravel()
        fig = axes[0].figure

    omega_factor = {
        "deg": np.degrees(1.0),
        "mrad": 1e3,
        "rad": 1.0
    }[omega_unit]

    xy, unit = _coords(tensor.positions, sampling)
    titles = dict(_TITLES, **(titles or {}))
    clims = clims or {}

    for ax, name in zip(axes, components, strict=False):
        raw = tensor.component(name)

        if name == "omega":
            vals = raw * omega_factor
            cbar_label = f"rotation ({omega_unit})"
        else:
            vals = raw * (100.0 if percent else 1.0)
            cbar_label = "strain (%)" if percent else "strain"

        ok = np.isfinite(vals)
        clim = clims.get(name) or symmetric_limits(
            vals[ok], clim_percentile
        )

        if kind == "scatter":
            m = ax.scatter(
                xy[ok, 0], xy[ok, 1],
                c=vals[ok],
                s=s,
                cmap=cmap,
                vmin=clim[0],
                vmax=clim[1],
                edgecolors="none"
            )

        elif kind == "grid":
            grid, xi, yi = interpolate_to_grid(
                tensor.positions, vals, step=step
            )
            k = 1.0 if sampling is None else sampling

            m = ax.imshow(
                grid,
                cmap=cmap,
                vmin=clim[0],
                vmax=clim[1],
                origin="upper",
                interpolation="nearest",
                extent=(
                    xi[0] * k, xi[-1] * k,
                    yi[-1] * k, yi[0] * k
                )
            )

        else:
            raise ValueError("kind must be 'scatter' or 'grid'")

        _style_map_axes(ax, unit, show_axes, titles[name])

        # Bold mathematical symbols in panel titles.
        ax.title.set_math_fontfamily("dejavusans")
        with plt.rc_context({"mathtext.default": "bf"}):
            ax.set_title(
                titles[name],
                fontsize=title_size,
                fontweight="bold",
                pad=14
            )

        for label in (ax.xaxis.label, ax.yaxis.label):
            label.set_fontsize(label_size)
            label.set_fontweight("bold")

        ax.tick_params(axis="both", labelsize=tick_size)

        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_fontweight("bold")

        # Horizontal and vertical colour bars use the same text style.
        cbar = add_colorbar(
            m, ax, cbar_label,
            location=colorbar_location
        )
        cbar.set_label(
            cbar_label,
            fontsize=label_size,
            fontweight="bold",
            labelpad=10
        )
        cbar.ax.tick_params(axis="both", labelsize=tick_size)

        for text in (
            cbar.ax.get_xticklabels()
            + cbar.ax.get_yticklabels()
        ):
            text.set_fontweight("bold")

        for panel in (ax, cbar.ax):
            for axis in (panel.xaxis, panel.yaxis):
                offset = axis.get_offset_text()
                offset.set_fontsize(tick_size)
                offset.set_fontweight("bold")

        if scalebar_nm is not None:
            existing_text = {id(text) for text in ax.findobj(Text)}

            add_scalebar(
                ax,
                sampling,
                None if scalebar_nm == "auto" else scalebar_nm,
                data_units=unit,
                color="black"
            )

            for text in ax.findobj(Text):
                if id(text) not in existing_text:
                    text.set_fontsize(tick_size)
                    text.set_fontweight("bold")

    fig.tight_layout(pad=2.0, w_pad=3.0)

    return fig, axes


def plot_strain_comparison(positions, values, reference_values, *,
                           labels=("Method 2", "Method 1"), sampling=None,
                           kind="scatter", step=2.0, percent=True, cmap="RdBu_r",
                           clim=None, colorbar_location="bottom", s=6,
                           figsize=None, title_size=20, label_size=18,
                           tick_size=16):
    """Two strain maps with shared colour limits and larger, bold text.

    Parameters
    ----------
    positions : array_like
        (N, 2) column positions.
    values, reference_values : array_like
        (N,) per-column values of the two maps.
    labels : (str, str)
        Panel titles.
    sampling, kind, step, percent, cmap, colorbar_location, s
        As in plot_strain_tensor.
    clim : (float, float), optional
        Shared limits; default symmetric at the 98th percentile of both.
    title_size, label_size, tick_size : float
        Font sizes for titles, axis/colour-bar labels and tick labels.

    Returns
    -------
    fig, axes
    """
    k = 100.0 if percent else 1.0
    a = np.asarray(values, dtype=float) * k
    b = np.asarray(reference_values, dtype=float) * k

    both = np.concatenate([
        a[np.isfinite(a)],
        b[np.isfinite(b)]
    ])

    if clim is None:
        clim = symmetric_limits(both)

    if figsize is None:
        figsize = (
            11,
            5.5 * min(data_aspect(positions), 1.6) + 1.5
        )

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    xy, unit = _coords(positions, sampling)

    for ax, vals, label in zip(axes, (a, b), labels, strict=True):
        ok = np.isfinite(vals)

        if kind == "scatter":
            m = ax.scatter(
                xy[ok, 0], xy[ok, 1],
                c=vals[ok],
                s=s,
                cmap=cmap,
                vmin=clim[0],
                vmax=clim[1],
                edgecolors="none"
            )

        elif kind == "grid":
            grid, xi, yi = interpolate_to_grid(
                positions, vals, step=step
            )
            f = 1.0 if sampling is None else sampling

            m = ax.imshow(
                grid,
                cmap=cmap,
                vmin=clim[0],
                vmax=clim[1],
                interpolation="nearest",
                extent=(
                    xi[0] * f, xi[-1] * f,
                    yi[-1] * f, yi[0] * f
                )
            )

        else:
            raise ValueError("kind must be 'scatter' or 'grid'")

        _style_map_axes(ax, unit, True, label)

        ax.set_title(
            label,
            fontsize=title_size,
            fontweight="bold",
            pad=14
        )

        for axis_label in (ax.xaxis.label, ax.yaxis.label):
            axis_label.set_fontsize(label_size)
            axis_label.set_fontweight("bold")

        ax.tick_params(axis="both", labelsize=tick_size)

        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_fontweight("bold")

        cbar_label = "strain (%)" if percent else "strain"
        cbar = add_colorbar(
            m, ax, cbar_label,
            location=colorbar_location
        )
        cbar.set_label(
            cbar_label,
            fontsize=label_size,
            fontweight="bold",
            labelpad=10
        )
        cbar.ax.tick_params(axis="both", labelsize=tick_size)

        for text in (
            cbar.ax.get_xticklabels()
            + cbar.ax.get_yticklabels()
        ):
            text.set_fontweight("bold")

        for panel in (ax, cbar.ax):
            for coordinate_axis in (panel.xaxis, panel.yaxis):
                offset = coordinate_axis.get_offset_text()
                offset.set_fontsize(tick_size)
                offset.set_fontweight("bold")

    fig.tight_layout(pad=2.0, w_pad=3.0)

    return fig, axes


def plot_profiles(profiles, *, sampling=None, axis="y", percent=True, ax=None,
                  ylabel="strain", errorbars="std", title=None,
                  title_size=20, label_size=18, tick_size=16,
                  legend_size=14):
    """Plot binned profiles with larger, bold labels, ticks and legend.

    Parameters
    ----------
    profiles : dict
        {label: profile_dict}.
    sampling : float, optional
        Pixel size in nm/px.
    axis : {"x", "y"}, default "y"
        Used for the horizontal axis label.
    percent : bool, default True
    ax : matplotlib.axes.Axes, optional
    ylabel : str, default "strain"
    errorbars : {"std", "sem", None}, default "std"
        Shaded standard deviation, standard error of the mean, or none.
    title : str, optional

    Returns
    -------
    fig, ax
    """
    fig, ax = get_axes(ax, figsize=(10, 6))

    k = 100.0 if percent else 1.0
    f = 1.0 if sampling is None else sampling

    for label, prof in profiles.items():
        x = np.asarray(prof["center"]) * f
        y = np.asarray(prof["mean"]) * k

        line, = ax.plot(
            x, y,
            marker="o",
            ms=3,
            label=label
        )

        if errorbars:
            e = np.asarray(prof[errorbars]) * k
            ax.fill_between(
                x, y - e, y + e,
                color=line.get_color(),
                alpha=0.2,
                linewidth=0
            )

    ax.axhline(0, color="k", lw=0.8, ls=":")

    ax.set_xlabel(
        f"{axis} ({'nm' if sampling is not None else 'px'})",
        fontsize=label_size,
        fontweight="bold",
        labelpad=10
    )
    ax.set_ylabel(
        f"{ylabel} (%)" if percent else ylabel,
        fontsize=label_size,
        fontweight="bold",
        labelpad=10
    )

    ax.tick_params(
        axis="both", which="both",
        labelsize=tick_size
    )

    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontweight("bold")

    for coordinate_axis in (ax.xaxis, ax.yaxis):
        offset = coordinate_axis.get_offset_text()
        offset.set_fontsize(tick_size)
        offset.set_fontweight("bold")

    ax.grid(True, alpha=0.3)

    ax.legend(
        prop={"size": legend_size, "weight": "bold"},
        framealpha=0.9
    )

    if title:
        ax.set_title(
            title,
            fontsize=title_size,
            fontweight="bold",
            pad=14
        )

    fig.tight_layout(pad=1.5)

    return fig, ax