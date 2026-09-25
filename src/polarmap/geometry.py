"""Small geometric helpers shared by several modules."""

from __future__ import annotations

import numpy as np

__all__ = ["inside_image", "in_rectangle"]


def inside_image(positions, shape, margin=0.0):
    """Boolean mask of positions strictly inside an image minus a margin.

    Parameters
    ----------
    positions : array_like
        ``(N, 2)`` positions ``(x, y)`` in pixels.
    shape : tuple of int
        Image shape ``(height, width)``.
    margin : float, default 0
        Distance from each border, in pixels, that is excluded.

    Returns
    -------
    numpy.ndarray
        ``(N,)`` boolean mask, ``True`` where
        ``margin < x < width - margin`` and ``margin < y < height - margin``.
    """
    pos = np.asarray(positions, dtype=float).reshape(-1, 2)
    h, w = shape[:2]
    return ((pos[:, 0] > margin) & (pos[:, 0] < w - margin)
            & (pos[:, 1] > margin) & (pos[:, 1] < h - margin))


def in_rectangle(positions, x_range, y_range):
    """Boolean mask of positions inside a closed rectangle.

    Parameters
    ----------
    positions : array_like
        ``(N, 2)`` positions ``(x, y)`` in pixels.
    x_range, y_range : (float, float)
        Inclusive ``(min, max)`` bounds in pixels. To specify a region in
        nanometres, divide by the sampling first.

    Returns
    -------
    numpy.ndarray
        ``(N,)`` boolean mask.
    """
    pos = np.asarray(positions, dtype=float).reshape(-1, 2)
    x0, x1 = sorted(x_range)
    y0, y1 = sorted(y_range)
    return ((pos[:, 0] >= x0) & (pos[:, 0] <= x1)
            & (pos[:, 1] >= y0) & (pos[:, 1] <= y1))
