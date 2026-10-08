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
                       s=6, show_axes=True, scalebar_nm=None, titles=None):
    """Multi-panel figure of strain-tensor components (Method 2).

    Parameters
    ----------
    tensor : polarmap.strain.TensorStrain
    kind : {"scatter", "grid"}, default "scatter"
        ``"scatter"``: atom-resolved (no interpolation); ``"grid"``: values
        interpolated linearly onto a grid with ``step`` px spacing, for display
        only.
    sampling : float, optional
        nm/px; axes in nm when given.
    components : sequence of str, default all four
        Any of ``"exx"``, ``"eyy"``, ``"exy"``, ``"omega"``.
    percent : bool, default True
        Strain components in %.
    omega_unit : {"deg", "mrad", "rad"}, default "deg"
    step : float, default 2
        Grid step for ``kind="grid"``.
    cmap : str, default "RdBu_r"
    clim_percentile : float, default 98
        Symmetric colour limits per panel at this percentile of the absolute values.
    clims : dict, optional
        Explicit ``{component: (vmin, vmax)}`` limits in displayed units.
    colorbar_location : str, default "bottom"
        Horizontal colour bars under each panel keep multi-panel figures
        compact.
    axes : sequence of Axes, optional
    figsize : tuple, optional
    s : float, default 6
        Marker size for ``kind="scatter"``.
    show_axes : bool, default True
    scalebar_nm : float or "auto", optional
    titles : dict, optional
        Override panel titles, ``{component: title}``.

    Returns
    -------
    fig, axes
    """
    components = list(components)
    if axes is None:
        if figsize is None:
            panel = 4.4
            extra = 1.5 if show_axes else 1.0        # title, labels, colour bar
            figsize = (panel * len(components),
                       panel * min(data_aspect(tensor.positions), 1.6) + extra)
        fig, axes = plt.subplots(1, len(components), figsize=figsize)
        axes = np.atleast_1d(axes)
    else:
        fig = np.atleast_1d(axes)[0].figure
    omega_factor = {"deg": np.degrees(1.0), "mrad": 1e3, "rad": 1.0}[omega_unit]
    xy, unit = _coords(tensor.positions, sampling)
    titles = dict(_TITLES, **(titles or {}))
    clims = clims or {}

    for ax, name in zip(axes, components, strict=False):
        raw = tensor.component(name)
        if name == "omega":
            vals, cbar_label = raw * omega_factor, f"rotation ({omega_unit})"
        else:
            vals = raw * (100.0 if percent else 1.0)
            cbar_label = "strain (%)" if percent else "strain"
        ok = np.isfinite(vals)
        clim = clims.get(name) or symmetric_limits(vals[ok], clim_percentile)
        if kind == "scatter":
            m = ax.scatter(xy[ok, 0], xy[ok, 1], c=vals[ok], s=s, cmap=cmap,
                           vmin=clim[0], vmax=clim[1], edgecolors="none")
        elif kind == "grid":
            grid, xi, yi = interpolate_to_grid(tensor.positions, vals, step=step)
            k = 1.0 if sampling is None else sampling
            m = ax.imshow(grid, cmap=cmap, vmin=clim[0], vmax=clim[1],
                          origin="upper", interpolation="nearest",
                          extent=(xi[0] * k, xi[-1] * k, yi[-1] * k, yi[0] * k))
        else:
            raise ValueError("kind must be 'scatter' or 'grid'")
        _style_map_axes(ax, unit, show_axes, titles[name])
        add_colorbar(m, ax, cbar_label, location=colorbar_location)
        if scalebar_nm is not None:
            add_scalebar(ax, sampling, None if scalebar_nm == "auto" else scalebar_nm,
                         data_units=unit, color="black")
    fig.tight_layout()
    return fig, axes


def plot_strain_comparison(positions, values, reference_values, *,
                           labels=("Method 2", "Method 1"), sampling=None,
                           kind="scatter", step=2.0, percent=True, cmap="RdBu_r",
                           clim=None, colorbar_location="bottom", s=6,
                           figsize=None):
    """Two strain maps side by side with shared colour limits.

    Parameters
    ----------
    positions : array_like
        ``(N, 2)`` column positions.
    values, reference_values : array_like
        ``(N,)`` per-column values of the two maps (e.g. ``eyy`` of Method 2 and
        the projection strain of Method 1).
    labels : (str, str)
    sampling, kind, step, percent, cmap, colorbar_location, s
        As in :func:`plot_strain_tensor`.
    clim : (float, float), optional
        Shared limits; default symmetric at the 98th percentile of both.

    Returns
    -------
    fig, axes
    """
    k = 100.0 if percent else 1.0
    a = np.asarray(values, dtype=float) * k
    b = np.asarray(reference_values, dtype=float) * k
    both = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
    clim = clim or symmetric_limits(both)
    if figsize is None:
        figsize = (11, 5.5 * min(data_aspect(positions), 1.6) + 1.5)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    xy, unit = _coords(positions, sampling)
    for ax, vals, label in zip(axes, (a, b), labels, strict=True):
        ok = np.isfinite(vals)
        if kind == "scatter":
            m = ax.scatter(xy[ok, 0], xy[ok, 1], c=vals[ok], s=s, cmap=cmap,
                           vmin=clim[0], vmax=clim[1], edgecolors="none")
        elif kind == "grid":
            grid, xi, yi = interpolate_to_grid(positions, vals, step=step)
            f = 1.0 if sampling is None else sampling
            m = ax.imshow(grid, cmap=cmap, vmin=clim[0], vmax=clim[1],
                          interpolation="nearest",
                          extent=(xi[0] * f, xi[-1] * f, yi[-1] * f, yi[0] * f))
        else:
            raise ValueError("kind must be 'scatter' or 'grid'")
        _style_map_axes(ax, unit, True, label)
        add_colorbar(m, ax, "strain (%)" if percent else "strain",
                     location=colorbar_location)
    fig.tight_layout()
    return fig, axes


def plot_profiles(profiles, *, sampling=None, axis="y", percent=True, ax=None,
                  ylabel="strain", errorbars="std", title=None):
    """Plot one or more binned profiles (:func:`polarmap.strain.binned_profile`).

    Parameters
    ----------
    profiles : dict
        ``{label: profile_dict}``.
    sampling : float, optional
        nm/px; the abscissa is in nm when given.
    axis : {"x", "y"}, default "y"
        Only used for the axis label.
    percent : bool, default True
    ax : matplotlib.axes.Axes, optional
    ylabel : str, default "strain"
    errorbars : {"std", "sem", None}, default "std"
        Shaded band: standard deviation within each band, standard error of the
        mean, or none.
    title : str, optional

    Returns
    -------
    fig, ax
    """
    fig, ax = get_axes(ax, figsize=(8, 4.5))
    k = 100.0 if percent else 1.0
    f = 1.0 if sampling is None else sampling
    for label, prof in profiles.items():
        x = np.asarray(prof["center"]) * f
        y = np.asarray(prof["mean"]) * k
        line, = ax.plot(x, y, marker="o", ms=3, label=label)
        if errorbars:
            e = np.asarray(prof[errorbars]) * k
            ax.fill_between(x, y - e, y + e, color=line.get_color(), alpha=0.2,
                            linewidth=0)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel(f"{axis} ({'nm' if sampling else 'px'})")
    ax.set_ylabel(f"{ylabel} (%)" if percent else ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title)
    return fig, ax
