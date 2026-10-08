"""Shared plotting helpers: image display, scale bars, colour bars, colour wheel."""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from ..statistics import ANGLE_CONVENTIONS

__all__ = [
    "CYCLIC_CMAP",
    "get_axes",
    "show_image",
    "nice_length",
    "add_scalebar",
    "add_colorbar",
    "add_colorwheel",
    "symmetric_limits",
]


def _default_cyclic_cmap():
    try:
        import colorcet  # noqa: F401  (registers the 'cet_*' colour maps)
        if "cet_colorwheel" in matplotlib.colormaps:
            return "cet_colorwheel"
    except ImportError:
        pass
    return "hsv"


#: Cyclic colour map used for orientations (colorcet's perceptually uniform
#: ``cet_colorwheel`` if colorcet is installed, otherwise Matplotlib's ``hsv``).
CYCLIC_CMAP = _default_cyclic_cmap()


def get_axes(ax=None, figsize=(7, 7)):
    """Return ``(fig, ax)``, creating a new figure if ``ax`` is ``None``."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def show_image(ax, image, *, dim=False, keep_contrast=0.55, wash=0.35,
               cmap="gray", percentiles=(1.0, 99.0), extent=None):
    """Display a grey-scale image with robust contrast limits.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    image : array_like
    dim : bool, default False
        Reduce the contrast and wash the image towards white so that overlaid
        arrows or colour maps remain visible over both bright and dark columns.
    keep_contrast, wash : float
        Strength of the dimming (only with ``dim=True``).
    cmap : str, default "gray"
    percentiles : (float, float), default (1, 99)
        Intensity percentiles mapped to black and white.
    extent : sequence, optional
        Passed to ``imshow`` (default: pixel coordinates).
    """
    image = np.asarray(image, dtype=float)
    lo, hi = np.nanpercentile(image, percentiles)
    if dim:
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo) / max(keep_contrast, 1e-3)
        lo, hi = mid - half, mid + half
    im = ax.imshow(image, cmap=cmap, vmin=lo, vmax=hi, extent=extent,
                   interpolation="nearest")
    if dim and wash > 0:
        ax.imshow(np.ones_like(image), cmap="gray", vmin=0, vmax=1, alpha=wash,
                  extent=extent)
    return im


def nice_length(max_length):
    """Largest value of the form 1, 2 or 5 x 10^k not exceeding ``max_length``."""
    if not max_length > 0:
        raise ValueError("max_length must be positive")
    exponent = np.floor(np.log10(max_length))
    for mantissa in (5, 2, 1):
        value = mantissa * 10 ** exponent
        if value <= max_length:
            return float(value)
    return float(10 ** exponent)


def add_scalebar(ax, sampling, length_nm=None, *, data_units="px",
                 location="lower right", color="white", fontsize=10,
                 thickness=None, label=None, frameon=False, pad=0.5):
    """Add a scale bar to an axes showing an image.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    sampling : float
        Pixel size in nm/px.
    length_nm : float, optional
        Bar length; by default a round value close to 20 % of the axes width.
    data_units : {"px", "nm"}, default "px"
        Units of the axes data coordinates (``"px"`` for plain ``imshow``,
        ``"nm"`` when an ``extent`` in nm was used).
    location : str, default "lower right"
        Any Matplotlib legend location.
    color : str, default "white"
    fontsize : float, default 10
    thickness : float, optional
        Bar thickness in data units; default 1 % of the axes height.
    label : str, optional
        Text; default e.g. ``"2 nm"``.
    frameon : bool, default False
    pad : float, default 0.5

    Returns
    -------
    mpl_toolkits.axes_grid1.anchored_artists.AnchoredSizeBar
    """
    if data_units not in ("px", "nm"):
        raise ValueError("data_units must be 'px' or 'nm'")
    per_nm = 1.0 / sampling if data_units == "px" else 1.0
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    width_nm = abs(x1 - x0) / per_nm
    if length_nm is None:
        length_nm = nice_length(0.2 * width_nm)
    if thickness is None:
        thickness = 0.01 * abs(y1 - y0)
    if label is None:
        label = f"{length_nm:g} nm"
    bar = AnchoredSizeBar(ax.transData, length_nm * per_nm, label, location,
                          pad=pad, color=color, frameon=frameon,
                          size_vertical=thickness,
                          fontproperties={"size": fontsize})
    ax.add_artist(bar)
    return bar


def add_colorbar(mappable, ax, label=None, *, location="right", size="4%",
                 pad=None, ticks=None, **kwargs):
    """Add a colour bar that matches the size of ``ax``.

    The colour bar gets its own axes appended to ``ax`` (via
    ``make_axes_locatable``), so it is exactly as tall (or, at the bottom, as
    wide) as the image instead of the whole figure. Call it after the axes
    labels are set so that the automatic gap leaves room for them.

    Parameters
    ----------
    mappable : matplotlib.cm.ScalarMappable
        The image, scatter or collection to describe.
    ax : matplotlib.axes.Axes
    label : str, optional
    location : {"right", "left", "bottom", "top"}, default "right"
        ``"bottom"`` gives a horizontal bar under the image, which saves space
        in multi-panel figures.
    size : str, default "4%"
        Thickness relative to the axes.
    pad : float, optional
        Gap in inches; by default just enough for the tick labels and axis
        label on that side of ``ax``.
    ticks : sequence, optional
    **kwargs
        Passed to ``Figure.colorbar``.

    Returns
    -------
    matplotlib.colorbar.Colorbar
    """
    if location not in ("right", "left", "bottom", "top"):
        raise ValueError("location must be 'right', 'left', 'bottom' or 'top'")
    if pad is None:
        pad = _auto_pad(ax, location)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(location, size=size, pad=pad)
    orientation = "horizontal" if location in ("bottom", "top") else "vertical"
    cbar = ax.figure.colorbar(mappable, cax=cax, orientation=orientation,
                              ticks=ticks, **kwargs)
    if location == "top":
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
    if location == "left":
        cax.yaxis.set_ticks_position("left")
        cax.yaxis.set_label_position("left")
    if label:
        cbar.set_label(label)
    return cbar


def _auto_pad(ax, location):
    """Gap (inches) between ``ax`` and an appended colour bar."""
    visible = ax.axison
    if location == "bottom":
        ticks = visible and ax.xaxis.get_visible()
        return 0.1 + (0.3 if ticks else 0.0) + (0.2 if ticks and ax.get_xlabel() else 0.0)
    if location == "left":
        ticks = visible and ax.yaxis.get_visible()
        return 0.1 + (0.4 if ticks else 0.0) + (0.25 if ticks and ax.get_ylabel() else 0.0)
    if location == "top":
        return 0.1 + (0.3 if ax.get_title() else 0.0)
    return 0.08


def data_aspect(positions):
    """Height / width of the bounding box of a set of positions (>= 0.05)."""
    pos = np.asarray(positions, dtype=float).reshape(-1, 2)
    pos = pos[np.all(np.isfinite(pos), axis=1)]
    if len(pos) < 2:
        return 1.0
    w, h = np.ptp(pos[:, 0]), np.ptp(pos[:, 1])
    return float(np.clip(h / w, 0.05, 20.0)) if w > 0 else 1.0


def add_colorwheel(ax, cmap=None, convention="cartesian",
                   bounds=(0.78, 0.78, 0.2, 0.2), label="direction"):
    """Add a round colour-wheel key for orientation colour coding.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes the wheel is inset into.
    cmap : str, optional
        Cyclic colour map; default :data:`CYCLIC_CMAP`.
    convention : {"cartesian", "image"}, default "cartesian"
        Must match the angles that were colour coded (see
        :mod:`polarmap.statistics`): with ``"cartesian"``, +90 deg is drawn at
        the top of the wheel; with ``"image"``, at the bottom.
    bounds : (x0, y0, width, height), default (0.78, 0.78, 0.2, 0.2)
        Position in axes-fraction coordinates.
    label : str, default "direction"

    Returns
    -------
    matplotlib.projections.polar.PolarAxes
    """
    if convention not in ANGLE_CONVENTIONS:
        raise ValueError(f"convention must be one of {ANGLE_CONVENTIONS}")
    cmap = CYCLIC_CMAP if cmap is None else cmap
    try:
        wax = ax.inset_axes(bounds, projection="polar")
    except (TypeError, ValueError):     # very old Matplotlib: place by figure coords
        fig = ax.figure
        box = ax.get_position()
        rect = (box.x0 + bounds[0] * box.width, box.y0 + bounds[1] * box.height,
                bounds[2] * box.width, bounds[3] * box.height)
        wax = fig.add_axes(rect, projection="polar")
    wax.set_theta_zero_location("E")
    wax.set_theta_direction(1 if convention == "cartesian" else -1)
    theta = np.linspace(-np.pi, np.pi, 361)
    radius = np.array([0.55, 1.0])
    values = np.degrees(theta)[:-1][None, :]
    wax.pcolormesh(theta, radius, values, cmap=cmap, vmin=-180, vmax=180,
                   shading="flat")
    wax.set_yticks([])
    wax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    wax.set_xticklabels(["0°", "90°", "±180°", "-90°"], fontsize=7)
    wax.grid(False)
    wax.spines["polar"].set_visible(False)
    wax.set_title(label, fontsize=8, pad=2)
    return wax


def symmetric_limits(values, percentile=98.0):
    """Symmetric colour limits ``(-v, v)`` with ``v`` the given percentile of ``abs(values)``."""
    vals = np.abs(np.asarray(values, dtype=float))
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -1.0, 1.0
    v = float(np.percentile(vals, percentile))
    v = v if v > 0 else float(vals.max()) or 1.0
    return -v, v
