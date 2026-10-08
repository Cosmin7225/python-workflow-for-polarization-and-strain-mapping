"""Tests for polarmap.synthetic: the ground truth itself must be right."""

import numpy as np
import pytest
from scipy.spatial import KDTree

from polarmap import synthetic


def test_render_columns_peak_and_integral():
    img = synthetic.render_columns([[20.0, 15.0]], 2.0, 1.5, (40, 40), background=0.1)
    assert img[15, 20] == pytest.approx(2.1)
    assert (img - 0.1).sum() == pytest.approx(2.0 * 2 * np.pi * 1.5 ** 2, rel=1e-6)
    outside = synthetic.render_columns([[-100.0, -100.0]], 1.0, 1.0, (10, 10))
    assert np.all(outside == 0)


def test_add_noise_is_reproducible():
    img = np.full((30, 30), 0.5)
    a = synthetic.add_noise(img, dose=100, gaussian_sigma=0.01, seed=3)
    b = synthetic.add_noise(img, dose=100, gaussian_sigma=0.01, seed=3)
    np.testing.assert_array_equal(a, b)
    assert abs(a.mean() - 0.5) < 0.02 and a.std() > 0.03


def test_perovskite_100_ground_truth():
    s = synthetic.perovskite_100(shape=(150, 160), a=25.0, displacement=(0.5, -1.0),
                                 rotation_deg=5.0)
    np.testing.assert_allclose(s.displacement, np.broadcast_to([0.5, -1.0], s.displacement.shape))
    assert np.linalg.norm(s.v1) == pytest.approx(25.0)
    assert np.dot(s.v1, s.v2) == pytest.approx(0.0, abs=1e-9)
    # each B ideal position is the centre of four A sites
    all_a = s.reference
    d, _ = KDTree(all_a).query(s.target_ideal, k=4)
    interior = d[:, 3] < 18
    np.testing.assert_allclose(d[interior], 25.0 / np.sqrt(2), atol=1e-9)
    field = synthetic.perovskite_100(displacement=lambda xy: np.c_[xy[:, 0] * 0, xy[:, 1] * 0 + 1])
    np.testing.assert_allclose(field.displacement[:, 1], 1.0)
    with pytest.raises(ValueError):
        synthetic.perovskite_100(displacement=lambda xy: np.zeros(3))


def test_strained_lattice_bonds_follow_F():
    F = np.array([[1.02, 0.01], [0.0, 0.97]])
    lat = synthetic.strained_lattice((200, 200), a=16.0, F=F)
    tree = KDTree(lat.positions)
    centre = np.argmin(np.linalg.norm(lat.positions - 100, axis=1))
    for v in (lat.v1, lat.v2):
        d, j = tree.query(lat.positions[centre] + F @ v)
        assert d < 1e-9
    exx, eyy, exy, om = lat.true_strain
    assert exx[0] == pytest.approx(0.02) and eyy[0] == pytest.approx(-0.03)


def test_superlattice_layers():
    lat = synthetic.superlattice((300, 100), a=16.0, c_values=(16.0, 17.6),
                                 cells_per_layer=4)
    assert set(np.unique(lat.labels)) >= {0, 1, 2}
    rows = np.unique(lat.positions[:, 1])
    gaps = np.round(np.diff(rows), 6)
    assert set(gaps) == {16.0, 17.6}
    fyy = lat.true_F[:, 1, 1]
    assert fyy.min() == pytest.approx(1.0) and fyy.max() == pytest.approx(1.1)
    rendered = synthetic.superlattice((120, 80), render=True, dose=200, seed=1)
    assert rendered.image.shape == (120, 80)
