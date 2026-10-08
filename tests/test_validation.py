"""Tests for polarmap.validation: VecMap reader and comparison metrics."""

import numpy as np
import pytest

from polarmap.displacement import DisplacementField
from polarmap.validation import (compare_displacement_fields, compare_maps,
                                 load_vecmap_csv, mutual_nearest_matches)

VECMAP_CSV = """x,y,u,v,magnitude,angle,A1x,A1y,A2x,A2y
207.402844,217.852257,0.001802,-1.292925,0.020687,270.079853,1,2,3,4
207.403164,192.049669,0.002436,-1.296027,0.020736,270.107712,1,2,3,4
183.000554,217.849009,0.000024,-1.297783,0.020765,270.001072,1,2,3,4
"""


def test_load_vecmap_csv(tmp_path):
    path = tmp_path / "vecmap.csv"
    path.write_text("﻿" + VECMAP_CSV + ",,,\n", encoding="utf-8")
    f = load_vecmap_csv(path)
    assert len(f) == 3
    np.testing.assert_allclose(f.target_xy[0], [207.402844, 217.852257])
    np.testing.assert_allclose(f.u[0], 0.001802)
    np.testing.assert_allclose(f.per_site["magnitude_nm"][2], 0.020765)
    # VecMap's own magnitude equals |(u, v)| * 0.016 nm/px
    np.testing.assert_allclose(f.magnitude_in(0.016, "nm"),
                               f.per_site["magnitude_nm"], rtol=2e-4)
    empty = tmp_path / "empty.csv"
    empty.write_text("x,y,u,v,magnitude,angle\n")
    with pytest.raises(ValueError, match="no data rows"):
        load_vecmap_csv(empty)


def test_mutual_nearest_matches():
    a = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    b = np.array([[0.3, 0.0], [10.2, 0.1], [10.9, 0.0], [50.0, 0.0]])
    ia, ib, d = mutual_nearest_matches(a, b, tolerance_px=1.5)
    np.testing.assert_array_equal(ia, [0, 1])
    np.testing.assert_array_equal(ib, [0, 1])
    assert d.max() < 1.5
    ia, _, _ = mutual_nearest_matches(a, b, tolerance_px=0.1)
    assert len(ia) == 0
    ia, ib, d = mutual_nearest_matches(np.empty((0, 2)), b)
    assert len(ia) == len(ib) == len(d) == 0


def test_compare_identical_and_offset_fields():
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, 200, 50), rng.uniform(0, 200, 50)
    u, v = rng.normal(0, 1, 50), rng.normal(0, 1, 50)
    f = DisplacementField(x, y, u, v)
    same = compare_displacement_fields(f, f, sampling=0.02)
    assert same["n_matched"] == 50
    for key in ("magnitude_bias_pm", "magnitude_rmse_pm", "vector_rmse_pm",
                "mean_abs_angle_deg"):
        assert same[key] == pytest.approx(0.0, abs=1e-9)
    shifted = DisplacementField(x, y, u + 0.1, v)
    cmp = compare_displacement_fields(shifted, f, sampling=0.02, match_on="reference")
    assert cmp["vector_rmse_pm"] == pytest.approx(2.0)          # 0.1 px * 20 pm/px
    np.testing.assert_allclose(cmp["delta_u_pm"], 2.0)
    assert len(cmp["magnitude_pm"]) == len(cmp["magnitude_other_pm"]) == 50
    with pytest.raises(ValueError, match="match_on"):
        compare_displacement_fields(f, f, 0.02, match_on="x")
    far = DisplacementField(x + 500, y, u, v)
    with pytest.raises(ValueError, match="no matching"):
        compare_displacement_fields(f, far, 0.02)


def test_compare_maps():
    a = np.array([[1.0, 2.0], [3.0, np.nan]])
    same = compare_maps(a, a)
    assert same["n"] == 3 and same["rmse"] == 0 and same["pearson_r"] == pytest.approx(1)
    off = compare_maps(a + 0.5, a)
    assert off["bias"] == pytest.approx(0.5) and off["std_difference"] == pytest.approx(0)
    masked = compare_maps(a, a + 1, mask=np.array([[True, True], [False, False]]))
    assert masked["n"] == 2 and masked["bias"] == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="same shape"):
        compare_maps(a, a.ravel())
    with pytest.raises(ValueError, match="fewer than two"):
        compare_maps(a, a, mask=np.zeros((2, 2), bool))
