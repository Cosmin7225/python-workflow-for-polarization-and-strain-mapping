"""Tests for polarmap.lattice: lattice vectors, graph, loop closure and BFS."""

import numpy as np
import pytest

from polarmap import synthetic
from polarmap.lattice import (DIRECTION_NAMES, bfs_lattice_indices,
                              build_lattice_graph, estimate_lattice_vectors,
                              estimate_pair_vector, find_bad_edges_by_closure)


def _same_up_to_sign(found, expected, rtol=5e-3):
    return (np.allclose(found, expected, rtol=0, atol=rtol * np.linalg.norm(expected))
            or np.allclose(found, -expected, rtol=0,
                           atol=rtol * np.linalg.norm(expected)))


# ------------------------------------------------------------ lattice vectors
@pytest.mark.parametrize("rotation", [0.0, 10.0, 37.0])
def test_estimate_lattice_vectors_square(rotation):
    lat = synthetic.strained_lattice((260, 260), a=16.0, rotation_deg=rotation)
    v1, v2 = estimate_lattice_vectors(lat.positions)
    found = sorted([v1, v2], key=lambda v: np.arctan2(v[1], v[0]))
    for expected in (lat.v1, lat.v2):
        assert any(_same_up_to_sign(f, expected) for f in found)


def test_estimate_lattice_vectors_rectangular_with_jitter():
    lat = synthetic.strained_lattice((300, 300), a=16.0, c=17.0, jitter=0.3,
                                     seed=1)
    v1, v2 = estimate_lattice_vectors(lat.positions)
    np.testing.assert_allclose(v1, [16.0, 0.0], atol=0.08)
    np.testing.assert_allclose(v2, [0.0, 17.0], atol=0.08)


@pytest.mark.parametrize("seed", range(5))
def test_axis_aligned_noisy_lattice_regression(seed):
    """Regression: bonds straddling the x axis must not average to ~zero.

    The notebook version folded bond vectors into the upper half-plane before
    averaging; with an x-aligned lattice and position noise, half of the +v1
    bonds were flipped to -v1 and the mean collapsed (|v1| ~ 1 px instead of
    16 px).
    """
    rng = np.random.default_rng(seed)
    i, j = np.meshgrid(np.arange(25), np.arange(25))
    pos = np.column_stack([16.0 * i.ravel(), 16.0 * j.ravel()])
    pos += rng.normal(0, 0.4, pos.shape)
    v1, v2 = estimate_lattice_vectors(pos)
    np.testing.assert_allclose(v1, [16.0, 0.0], atol=0.1)
    np.testing.assert_allclose(v2, [0.0, 16.0], atol=0.1)


def test_lattice_vector_order_and_signs():
    lat = synthetic.strained_lattice((300, 300), a=16.0, rotation_deg=-20.0)
    v1, v2 = estimate_lattice_vectors(lat.positions)
    assert v1[0] > 0 and v2[1] > 0
    assert abs(v1[0]) / np.linalg.norm(v1) >= abs(v2[0]) / np.linalg.norm(v2)
    assert v1[0] * v2[1] - v1[1] * v2[0] > 0      # right-handed in image axes


def test_estimate_lattice_vectors_warns_for_strongly_rectangular():
    lat = synthetic.strained_lattice((300, 300), a=12.0, c=20.0)
    with pytest.warns(UserWarning, match="only one lattice direction"):
        estimate_lattice_vectors(lat.positions)


def test_estimate_lattice_vectors_input_checks():
    with pytest.raises(ValueError):
        estimate_lattice_vectors(np.zeros((3, 2)))


def test_estimate_pair_vector(perov110):
    pv = estimate_pair_vector(perov110.reference, axis_hint=(0.1, 1.0))
    np.testing.assert_allclose(pv, perov110.v2, atol=1e-6)
    pv_up = estimate_pair_vector(perov110.reference, axis_hint=(0, -1))
    np.testing.assert_allclose(pv_up, -perov110.v2, atol=1e-6)
    with pytest.raises(ValueError):
        estimate_pair_vector(perov110.reference, axis_hint=(0, 0))


# ------------------------------------------------------------ lattice graph
def test_graph_on_perfect_lattice(square_points):
    pos, idx = square_points
    g = build_lattice_graph(pos, [10, 0], [0, 10])
    assert not g.bad_edges.any()
    interior = ((idx[:, 0] > 0) & (idx[:, 0] < 11) & (idx[:, 1] > 0) & (idx[:, 1] < 9))
    assert np.all(g.neighbor_idx[interior] >= 0)
    expected = np.array([[10, 0], [-10, 0], [0, 10], [0, -10]], dtype=float)
    np.testing.assert_allclose(g.neighbor_vec[interior], np.broadcast_to(
        expected, (interior.sum(), 4, 2)))
    # border columns lack exactly the outward bonds
    corner = np.flatnonzero((idx[:, 0] == 0) & (idx[:, 1] == 0))[0]
    assert list(g.neighbor_idx[corner] >= 0) == [True, False, True, False]
    s = g.summary()
    assert s["n_atoms"] == len(pos) and s["n_bonds_rejected"] == 0
    assert "Lattice graph" in g.report()


def test_graph_matches_true_neighbours_under_jitter():
    """Transverse-offset matching finds the in-line neighbour despite noise."""
    rng = np.random.default_rng(3)
    i, j = np.meshgrid(np.arange(20), np.arange(20))
    idx = np.column_stack([i.ravel(), j.ravel()])
    pos = 12.0 * idx + rng.normal(0, 1.0, idx.shape)           # 8 % jitter
    g = build_lattice_graph(pos, [12, 0], [0, 12])
    lookup = {tuple(k): n for n, k in enumerate(idx)}
    steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    wrong = 0
    for n, (a, b) in enumerate(idx):
        for d, (da, db) in enumerate(steps):
            true = lookup.get((a + da, b + db), -1)
            if g.neighbor_idx[n, d] != true:
                wrong += 1
    assert wrong == 0


def test_graph_under_shear_uses_perpendicular_offset():
    """In a sheared lattice with a distractor the in-line column is chosen."""
    pos = np.array([[0.0, 0.0], [10.0, 0.0], [8.0, 3.0]])
    g = build_lattice_graph(pos, [10.0, 0.0], [0.0, 10.0])
    # From column 0 along +v1: column 2 is axially closer (8 < 10) but 3 px
    # off-axis; column 1 is exactly in line and must win.
    assert g.neighbor_idx[0, 0] == 1


def test_closure_flags_inconsistent_bond(square_points):
    pos, idx = square_points
    g = build_lattice_graph(pos, [10, 0], [0, 10])
    corrupted = g.neighbor_idx.copy()
    lookup = {tuple(k): n for n, k in enumerate(idx)}
    a = lookup[(4, 4)]
    corrupted[a, 0] = lookup[(5, 5)]            # +v1 bond points to a wrong column
    bad = find_bad_edges_by_closure(corrupted)
    assert bad[a, 0]
    assert bad.sum() >= 2
    # bonds far from the corruption stay valid
    far = lookup[(9, 1)]
    assert not bad[far].any()


def test_graph_input_validation(square_points):
    pos, _ = square_points
    with pytest.raises(ValueError, match="parallel"):
        build_lattice_graph(pos, [10, 0], [20, 0])
    with pytest.raises(ValueError, match="non-zero"):
        build_lattice_graph(pos, [0, 0], [0, 10])
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        build_lattice_graph(pos.ravel(), [10, 0], [0, 10])


def test_direction_pair_and_describe(square_points):
    pos, _ = square_points
    g = build_lattice_graph(pos, [10, 0], [0, 10])
    assert g.direction_pair([0, 5]) == (2, 3)
    assert g.direction_pair([-1, 0.1]) == (1, 0)
    text = g.describe_column(0)
    assert "column 0" in text and all(name in text for name in DIRECTION_NAMES)
    np.testing.assert_allclose(g.v1, [10, 0])
    np.testing.assert_allclose(g.v2, [0, 10])


def test_clean_idx_masks_bad_edges(square_points):
    pos, _ = square_points
    g = build_lattice_graph(pos, [10, 0], [0, 10])
    g.bad_edges[5, 0] = True
    assert g.clean_idx[5, 0] == -1 and g.neighbor_idx[5, 0] >= 0


# ------------------------------------------------------------ BFS indexing
def test_bfs_recovers_integer_indices(square_points):
    pos, idx = square_points
    g = build_lattice_graph(pos, [10, 0], [0, 10])
    anchor = 37
    ind, reached = bfs_lattice_indices(g.clean_idx, anchor)
    assert reached.all()
    np.testing.assert_array_equal(ind, idx - idx[anchor])


def test_bfs_is_immune_to_accumulated_drift():
    """Rounding (r - r0)/a from a distant origin slips; BFS does not.

    The lattice spacing grows slowly across the image (a strain gradient).
    Rounding positions with the spacing measured near the origin assigns wrong
    indices far away; BFS through the bonds stays correct.
    """
    n = 60
    x = np.cumsum(np.r_[0, 10 * (1 + 0.004 * np.arange(n - 1))])   # 0 .. +24 %
    pos = np.column_stack([np.tile(x, 3), np.repeat([0.0, 10.0, 20.0], n)])
    true_i = np.tile(np.arange(n), 3)
    g = build_lattice_graph(pos, [10, 0], [0, 10], high_frac=1.6)
    ind, reached = bfs_lattice_indices(g.clean_idx, 0)
    assert reached.all()
    np.testing.assert_array_equal(ind[:, 0], true_i)
    rounded = np.round((pos[:, 0] - pos[0, 0]) / 10.0)
    assert np.any(rounded != true_i)        # the naive approach fails here


def test_bfs_disconnected_component():
    left = np.column_stack([np.repeat(np.arange(3) * 10.0, 3), np.tile(np.arange(3) * 10.0, 3)])
    right = left + [200.0, 0.0]
    g = build_lattice_graph(np.vstack([left, right]), [10, 0], [0, 10])
    ind, reached = bfs_lattice_indices(g.clean_idx, 0)
    assert reached[:9].all() and not reached[9:].any()
    assert np.isnan(ind[9:]).all()
    with pytest.raises(IndexError):
        bfs_lattice_indices(g.clean_idx, 99)
