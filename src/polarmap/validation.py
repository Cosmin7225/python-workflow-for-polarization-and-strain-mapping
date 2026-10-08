"""Quantitative comparison with other tools and with ground truth.

* :func:`load_vecmap_csv` reads the B-site displacement table exported by
  VecMap (Ma et al.) so that the two analyses can be compared column by
  column.
* :func:`mutual_nearest_matches` pairs two sets of positions robustly.
* :func:`compare_displacement_fields` reports bias, MAE and RMSE of the
  magnitudes, the vector RMSE and angular differences of matched vectors.
* :func:`compare_maps` reports the same kind of metrics for two scalar maps on
  a common grid or column set (e.g. a strain component against GPA or
  peak-pairs results).
"""

from __future__ import annotations

import csv

import numpy as np
from scipy.spatial import KDTree

from .displacement import DisplacementField
from .statistics import angular_difference

__all__ = [
    "load_vecmap_csv",
    "mutual_nearest_matches",
    "compare_displacement_fields",
    "compare_maps",
]


def load_vecmap_csv(path):
    """Read a VecMap B-site displacement CSV file.

    Only the first six entries of each data row are used:
    ``x, y, u, v, magnitude (nm), angle (deg)``, where ``(x, y)`` is the
    measured B-column position and ``(u, v)`` the displacement in pixels (image
    axes). The remaining entries (neighbouring reference columns) are ignored.

    Parameters
    ----------
    path : str or path-like

    Returns
    -------
    DisplacementField
        Anchored at the reference positions ``(x - u, y - v)``;
        ``per_site`` holds VecMap's own ``magnitude_nm`` and ``angle_vecmap_deg``.
    """
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh, skipinitialspace=True)
        next(reader, None)                                 # header
        for row in reader:
            values = [value.strip() for value in row]
            if len(values) < 6 or not values[0]:
                continue
            rows.append([float(v) for v in values[:6]])
    if not rows:
        raise ValueError(f"no data rows found in {path}")
    data = np.array(rows)
    xb, yb, u, v = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    return DisplacementField(xb - u, yb - v, u, v,
                             per_site={"magnitude_nm": data[:, 4],
                                       "angle_vecmap_deg": data[:, 5]})


def mutual_nearest_matches(xy_first, xy_second, tolerance_px=1.5):
    """Match points of two sets that are mutual nearest neighbours.

    A pair ``(i, j)`` is accepted only if ``j`` is the nearest point of the
    second set to ``i``, ``i`` is the nearest point of the first set to ``j``,
    and their distance is at most ``tolerance_px``.

    Returns
    -------
    first_indices, second_indices : numpy.ndarray
        Matched indices.
    distances : numpy.ndarray
        Matching distances in pixels.
    """
    a = np.asarray(xy_first, dtype=float).reshape(-1, 2)
    b = np.asarray(xy_second, dtype=float).reshape(-1, 2)
    if len(a) == 0 or len(b) == 0:
        empty = np.empty(0, dtype=int)
        return empty, empty, np.empty(0)
    dist, second = KDTree(b).query(a, k=1)
    _, back = KDTree(a).query(b, k=1)
    first = np.arange(len(a))
    ok = np.isfinite(dist) & (dist <= tolerance_px) & (back[second] == first)
    return first[ok], second[ok], dist[ok]


def compare_displacement_fields(field, other, sampling, *, tolerance_px=1.5,
                                match_on="target", convention="cartesian"):
    """Column-by-column comparison of two displacement fields.

    Parameters
    ----------
    field, other : DisplacementField
        The fields to compare (e.g. this workflow and VecMap). Use the same
        displacement sign convention for both.
    sampling : float
        Pixel size in nm/px (results are in pm).
    tolerance_px : float, default 1.5
        Matching tolerance.
    match_on : {"target", "reference"}, default "target"
        Match the measured target columns or the reference positions.
    convention : {"cartesian", "image"}, default "cartesian"
        Angle convention.

    Returns
    -------
    dict
        ``n_matched``, ``median_magnitude_pm`` (both fields),
        ``magnitude_bias_pm`` (mean of ``|d| - |d_other|``),
        ``magnitude_mae_pm``, ``magnitude_rmse_pm``, ``vector_rmse_pm``,
        ``median_abs_angle_deg``, ``mean_abs_angle_deg``, the matched indices
        ``index`` and ``index_other``, and per-site arrays ``magnitude_pm``,
        ``magnitude_other_pm``, ``delta_u_pm``, ``delta_v_pm``,
        ``delta_magnitude_pm`` and ``delta_angle_deg``.
    """
    if match_on == "target":
        xy_a, xy_b = field.target_xy, other.target_xy
    elif match_on == "reference":
        xy_a, xy_b = field.reference_xy, other.reference_xy
    else:
        raise ValueError("match_on must be 'target' or 'reference'")
    ia, ib, dist = mutual_nearest_matches(xy_a, xy_b, tolerance_px)
    if len(ia) == 0:
        raise ValueError("no matching columns found; check tolerance_px and "
                         "that both fields refer to the same image")
    k = sampling * 1e3
    du = (field.u[ia] - other.u[ib]) * k
    dv = (field.v[ia] - other.v[ib]) * k
    mag_a = field.magnitude[ia] * k
    mag_b = other.magnitude[ib] * k
    dmag = mag_a - mag_b
    dang = angular_difference(field.angle(convention)[ia],
                              other.angle(convention)[ib])
    return dict(
        n_matched=int(len(ia)),
        median_matching_distance_px=float(np.median(dist)),
        median_magnitude_pm=float(np.median(mag_a)),
        median_magnitude_other_pm=float(np.median(mag_b)),
        magnitude_bias_pm=float(np.mean(dmag)),
        magnitude_mae_pm=float(np.mean(np.abs(dmag))),
        magnitude_rmse_pm=float(np.sqrt(np.mean(dmag ** 2))),
        vector_rmse_pm=float(np.sqrt(np.mean(du ** 2 + dv ** 2))),
        median_abs_angle_deg=float(np.median(np.abs(dang))),
        mean_abs_angle_deg=float(np.mean(np.abs(dang))),
        index=ia, index_other=ib,
        magnitude_pm=mag_a, magnitude_other_pm=mag_b,
        delta_u_pm=du, delta_v_pm=dv, delta_magnitude_pm=dmag,
        delta_angle_deg=dang,
    )


def compare_maps(values, reference_values, mask=None):
    """Agreement metrics between two scalar maps sampled at the same points.

    Parameters
    ----------
    values, reference_values : array_like
        Arrays of identical shape (e.g. two strain grids, or per-column values);
        NaNs are ignored pairwise.
    mask : array_like of bool, optional
        Restrict the comparison to these points.

    Returns
    -------
    dict
        ``n`` (points compared), ``bias`` (mean difference), ``mae``, ``rmse``,
        ``std_difference`` and ``pearson_r``.
    """
    a = np.asarray(values, dtype=float)
    b = np.asarray(reference_values, dtype=float)
    if a.shape != b.shape:
        raise ValueError("both maps must have the same shape")
    ok = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        ok &= np.asarray(mask, dtype=bool)
    if ok.sum() < 2:
        raise ValueError("fewer than two points to compare")
    d = a[ok] - b[ok]
    r = (np.corrcoef(a[ok], b[ok])[0, 1]
         if a[ok].std() > 0 and b[ok].std() > 0 else np.nan)
    return dict(n=int(ok.sum()), bias=float(d.mean()),
                mae=float(np.abs(d).mean()), rmse=float(np.sqrt((d ** 2).mean())),
                std_difference=float(d.std()), pearson_r=float(r))
