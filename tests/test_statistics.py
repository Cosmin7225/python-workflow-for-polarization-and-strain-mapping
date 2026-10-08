"""Tests for polarmap.statistics: conventions, circular and robust statistics."""

import numpy as np
import pytest

from polarmap.displacement import DisplacementField
from polarmap.statistics import (angular_difference, circular_statistics,
                                 describe_displacements, displacement_angle,
                                 format_descriptors, local_outlier_mask,
                                 mad_outlier_mask)


def test_angle_conventions():
    # right, up (v < 0 in image axes), left, down
    u = np.array([1.0, 0.0, -1.0, 0.0])
    v = np.array([0.0, -1.0, 0.0, 1.0])
    np.testing.assert_allclose(displacement_angle(u, v, "cartesian"),
                               [0, 90, 180, -90])
    np.testing.assert_allclose(displacement_angle(u, v, "image"),
                               [0, -90, 180, 90])
    np.testing.assert_allclose(displacement_angle(u, v, degrees=False)[1], np.pi / 2)
    with pytest.raises(ValueError):
        displacement_angle(u, v, "polar")


def test_angular_difference_wraps():
    assert angular_difference(179.0, -179.0) == pytest.approx(-2.0)
    assert angular_difference(-179.0, 179.0) == pytest.approx(2.0)
    assert angular_difference(10.0, 350.0) == pytest.approx(20.0)
    d = angular_difference(np.linspace(-720, 720, 97), 0.0)
    assert np.all((d >= -180) & (d < 180))


def test_circular_statistics():
    around_180 = np.r_[np.full(50, 170.0), np.full(50, -170.0)]
    s = circular_statistics(around_180)
    assert abs(angular_difference(s["mean"], 180.0)) < 1e-9
    assert s["resultant_length"] == pytest.approx(np.cos(np.deg2rad(10)))
    # a naive arithmetic mean would give 0 deg here
    assert np.mean(around_180) == pytest.approx(0.0)
    uniform = circular_statistics(np.linspace(-180, 180, 360, endpoint=False))
    assert uniform["resultant_length"] < 1e-10
    empty = circular_statistics([np.nan])
    assert empty["n"] == 0 and np.isnan(empty["mean"])
    same = circular_statistics([42.0] * 5)
    assert same["std"] == pytest.approx(0.0, abs=1e-5)


def test_mad_outlier_mask():
    x = np.r_[np.zeros(20), 1.0, np.nan]
    x[:10] = 0.01
    both = mad_outlier_mask(x, 5)
    assert both[20] and not both[:20].any() and not both[21]
    low = np.r_[np.full(10, 1.0), np.full(10, 1.1), -50.0]
    assert mad_outlier_mask(low, 5)[-1]
    assert not mad_outlier_mask(low, 5, two_sided=False)[-1]   # lower tail kept
    assert not mad_outlier_mask(np.ones(10), 3).any()
    assert not mad_outlier_mask([np.nan, np.nan], 3).any()


def test_describe_displacements_rule():
    rng = np.random.default_rng(0)
    u = rng.normal(0, 0.05, 200)
    v = -1.3 + rng.normal(0, 0.05, 200)
    u[0], v[0] = 30.0, 30.0                     # a gross outlier
    f = DisplacementField(np.arange(200.0), np.zeros(200), u, v)
    d = describe_displacements(f, sampling=0.016)
    assert d["n"] == 199 and d["n_excluded"] == 1 and not d["kept"][0]
    assert d["median_pm"] == pytest.approx(1.3 * 16, rel=0.01)
    assert d["circular_mean_deg"] == pytest.approx(90, abs=1)
    text = format_descriptors(d)
    assert "n = 199" in text and "cartesian" in text
    everything = describe_displacements(f, sampling=0.016, n_mad=None)
    assert everything["n"] == 200 and everything["n_excluded"] == 0


def test_local_outlier_mask_keeps_coherent_structure():
    x, y = np.meshgrid(np.arange(30.0), np.arange(30.0))
    pos = np.column_stack([x.ravel(), y.ravel()])
    vals = np.where(pos[:, 1] < 5, 1.0, 0.0)            # a thin "layer"
    vals += np.random.default_rng(0).normal(0, 0.01, len(vals))
    vals[400] = 5.0                                      # one isolated spike
    local = local_outlier_mask(pos, vals, n_mad=5)
    assert local[400] and local.sum() == 1
    glob = mad_outlier_mask(vals, 5)
    assert glob[pos[:, 1] < 5].all()                     # global rule deletes the layer
    tiny = local_outlier_mask(pos[:5], vals[:5])
    assert tiny.shape == (5,)
