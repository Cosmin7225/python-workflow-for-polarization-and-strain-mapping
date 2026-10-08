"""Tests for polarmap.displacement against synthetic ground truth."""

import numpy as np
import pytest
from scipy.spatial import KDTree

from polarmap import synthetic
from polarmap.displacement import (DisplacementField, displacement_from_cage,
                                   displacement_from_offset,
                                   displacement_from_sublattices,
                                   displacement_to_polarization,
                                   find_complete_perovskite_cages,
                                   find_complete_reference_pairs,
                                   local_residual_maxima,
                                   measure_cage_displacement,
                                   measure_pair_displacement,
                                   predict_second_sublattice)


def _errors_vs_truth(field, synth):
    """Displacement error (px) of each measured vector vs the nearest true one."""
    _, idx = KDTree(synth.target_ideal).query(field.reference_xy)
    true = synth.displacement[idx]
    return np.column_stack([field.u, field.v]) - true


# ---------------------------------------------------------------- container
def test_displacement_field_basics(tmp_path):
    f = DisplacementField([0, 10], [0, 0], [3, 0], [4, -2],
                          per_site={"amp": np.array([1.0, 2.0])},
                          diagnostics={"note": "x"})
    assert len(f) == 2
    np.testing.assert_allclose(f.magnitude, [5, 2])
    np.testing.assert_allclose(f.magnitude_in(0.01, "pm"), [50, 20])
    np.testing.assert_allclose(f.target_xy, [[3, 4], [10, -2]])
    np.testing.assert_allclose(f.angle("cartesian"), [np.degrees(np.arctan2(-4, 3)), 90])
    sub = f.subset([False, True])
    assert len(sub) == 1 and sub.per_site["amp"][0] == 2.0 and sub.diagnostics["note"] == "x"
    flipped = f.with_sign(-1)
    np.testing.assert_allclose(flipped.u, [-3, 0])
    with pytest.raises(ValueError):
        f.with_sign(2)
    path = tmp_path / "f.csv"
    f.to_csv(path, sampling=0.02)
    back = DisplacementField.from_csv(path)
    np.testing.assert_allclose(back.u, f.u)
    np.testing.assert_allclose(back.y, f.y)


def test_displacement_field_validation():
    with pytest.raises(ValueError, match="same length"):
        DisplacementField([0, 1], [0], [0, 1], [0, 1])
    with pytest.raises(ValueError, match="per_site"):
        DisplacementField([0, 1], [0, 1], [0, 1], [0, 1], per_site={"a": [1]})


# ---------------------------------------------------------------- cages
def test_complete_cages_on_perfect_lattice(square_points):
    pos, idx = square_points
    centres, corners = find_complete_perovskite_cages(pos, [10, 0], [0, 10])
    assert len(centres) == 11 * 9
    np.testing.assert_allclose(centres, pos[corners].mean(axis=1))
    # every centre sits at the middle of a cell
    np.testing.assert_allclose(np.mod(centres - 20.0, 10.0), 5.0)


def test_missing_column_removes_its_four_cages(square_points):
    pos, idx = square_points
    keep = ~((idx[:, 0] == 5) & (idx[:, 1] == 5))
    centres, _ = find_complete_perovskite_cages(pos[keep], [10, 0], [0, 10])
    assert len(centres) == 11 * 9 - 4


def test_cage_tolerance_validation(square_points):
    pos, _ = square_points
    with pytest.raises(ValueError, match="tolerance"):
        find_complete_perovskite_cages(pos, [10, 0], [0, 10], tolerance=6.0)
    empty_c, empty_i = find_complete_perovskite_cages(pos[:3], [10, 0], [0, 10])
    assert empty_c.shape == (0, 2) and empty_i.shape == (0, 4)


def test_measure_cage_displacement_noise_free(perov100):
    s = perov100
    f = measure_cage_displacement(s.image, s.reference, s.v1, s.v2)
    # every complete cage (all four A corners inside the image) is measured;
    # B columns at the border, whose cages are incomplete, are rejected
    centres, _ = find_complete_perovskite_cages(s.reference, s.v1, s.v2)
    assert len(f) == len(centres) == 49
    err = _errors_vs_truth(f, s)
    assert np.abs(err).max() < 0.02                 # px
    assert f.diagnostics["residual"].shape == s.image.shape
    assert f.per_site["corner_indices"].shape == (len(f), 4)
    # the default direction convention: arrow from reference to B column
    assert np.median(f.v) == pytest.approx(-1.3, abs=0.02)
    flipped = measure_cage_displacement(s.image, s.reference, s.v1, s.v2, sign=-1)
    np.testing.assert_allclose(flipped.u, -f.u)


def test_measure_cage_displacement_noisy(perov100_noisy):
    s = perov100_noisy
    f = measure_cage_displacement(s.image, s.reference, s.v1, s.v2)
    err = _errors_vs_truth(f, s)
    rms = np.sqrt(np.mean(np.sum(err ** 2, axis=1)))
    assert rms < 0.15
    assert np.abs(err.mean(axis=0)).max() < 0.05     # no systematic bias


def test_nonpolar_control_gives_zero():
    """A centrosymmetric structure must give (almost) zero displacement."""
    s = synthetic.perovskite_100(shape=(200, 200), displacement=(0.0, 0.0),
                                 dose=400, noise_sigma=0.01, seed=11)
    f = measure_cage_displacement(s.image, s.reference, s.v1, s.v2)
    mean_disp = np.hypot(f.u.mean(), f.v.mean())
    assert mean_disp < 0.03                          # px: no spurious polarization
    assert np.median(f.magnitude) < 0.2              # per-cell noise floor


def test_domains_are_resolved():
    """Up-polarised left half, down-polarised right half."""
    def domains(xy):
        return np.where(xy[:, :1] < 100, [[0.0, -1.2]], [[0.0, 1.2]])
    s = synthetic.perovskite_100(shape=(200, 200), displacement=domains)
    f = measure_cage_displacement(s.image, s.reference, s.v1, s.v2)
    left, right = f.x < 90, f.x > 110
    assert np.all(f.angle()[left] > 80) and np.all(f.angle()[right] < -80)


def test_cage_options(perov100):
    s = perov100
    com = measure_cage_displacement(s.image, s.reference, s.v1, s.v2, refine="com")
    assert "amplitude" not in com.per_site
    assert np.median(com.v) == pytest.approx(-1.3, abs=0.3)
    anchor = measure_cage_displacement(s.image, s.reference, s.v1, s.v2,
                                       reference="anchor")
    assert np.all(anchor.per_site["corner_indices"] == -1)
    with pytest.raises(ValueError, match="reference"):
        measure_cage_displacement(s.image, s.reference, s.v1, s.v2, reference="x")
    with pytest.raises(ValueError, match="refine"):
        measure_cage_displacement(s.image, s.reference, s.v1, s.v2, refine="x")
    with pytest.raises(RuntimeError, match="No complete cages"):
        measure_cage_displacement(s.image, s.reference[:3], s.v1, s.v2)


# ---------------------------------------------------------------- pairs
def test_reference_pairs_geometry():
    ref = np.array([[0.0, 0.0], [0.0, 20.0], [0.0, 40.0], [30.0, 0.0]])
    ideal, pairs = find_complete_reference_pairs(ref, [0, 20], reference_fraction=0.25)
    np.testing.assert_allclose(ideal, [[0, 5], [0, 25]])
    np.testing.assert_array_equal(pairs, [[0, 1], [1, 2]])
    with pytest.raises(ValueError, match="between 0 and 1"):
        find_complete_reference_pairs(ref, [0, 20], reference_fraction=1.0)
    with pytest.raises(ValueError, match="non-zero"):
        find_complete_reference_pairs(ref, [0, 0])
    with pytest.raises(ValueError, match="tolerance"):
        find_complete_reference_pairs(ref, [0, 20], tolerance=11)


def test_measure_pair_displacement(perov110):
    s = perov110
    f = measure_pair_displacement(s.image, s.reference, s.v2)
    assert len(f) > 0.6 * len(s.target)
    err = _errors_vs_truth(f, s)
    assert np.abs(err).max() < 0.05
    assert set(f.per_site) >= {"pair_indices", "seeds", "amplitude", "ellipticity"}


def test_measure_pair_displacement_asymmetric_fraction():
    s = synthetic.perovskite_110(shape=(210, 220), displacement=(0.3, 0.5),
                                 reference_fraction=0.375)
    f = measure_pair_displacement(s.image, s.reference, s.v2,
                                  reference_fraction=0.375)
    err = _errors_vs_truth(f, s)
    assert np.abs(err).max() < 0.1


def test_local_residual_maxima():
    img = np.zeros((20, 20))
    img[12, 7] = 5.0
    seeds = local_residual_maxima(img, [[8, 10]], search_radius=3)
    np.testing.assert_array_equal(seeds, [[7, 12]])


# ---------------------------------------------------------------- point-set models
def test_from_sublattices_and_cage(square_points):
    A, _ = square_points
    shift = np.array([0.7, -0.4])
    B = predict_second_sublattice(A, [10, 0], [0, 10]) + shift
    f = displacement_from_sublattices(A, B, [10, 0], [0, 10])
    np.testing.assert_allclose(np.column_stack([f.u, f.v]),
                               np.broadcast_to(shift, (len(f), 2)))
    interior_B = B[(B[:, 0] < 125) & (B[:, 1] < 105)]
    g = displacement_from_cage(interior_B, A)
    np.testing.assert_allclose(g.u, 0.7, atol=1e-9)
    np.testing.assert_allclose(g.v, -0.4, atol=1e-9)


def test_from_offset_wurtzite_like():
    cations = np.column_stack([np.arange(5) * 10.0, np.zeros(5)])
    c_px, u_ref = 16.0, 3 / 8
    anions = cations + [0.0, u_ref * c_px] + [0.1, 0.25]
    f = displacement_from_offset(cations, anions, [0.0, u_ref * c_px])
    np.testing.assert_allclose(f.u, 0.1)
    np.testing.assert_allclose(f.v, 0.25)


def test_displacement_to_polarization_pbtio3():
    # 20.5 pm Ti shift, Z* = 7.1, a = 0.402 nm, c = 0.413 nm  ->  ~34.9 µC/cm^2
    p = displacement_to_polarization(20.5, 7.1, 0.402 * 0.402 * 0.413)
    assert p == pytest.approx(34.94, abs=0.05)
    assert displacement_to_polarization(0.0, 7.1, 0.066) == 0.0
