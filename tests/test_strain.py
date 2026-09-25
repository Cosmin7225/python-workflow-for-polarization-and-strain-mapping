"""Tests for polarmap.strain (Method 1 and Method 2) against exact ground truth."""

import numpy as np
import pytest

from polarmap import synthetic
from polarmap.geometry import in_rectangle, inside_image
from polarmap.lattice import build_lattice_graph
from polarmap.strain import (ProjectionStrain, ReferenceLattice, TensorStrain,
                             binned_profile, deformation_gradients,
                             fit_reference_lattice, interpolate_to_grid,
                             line_profile, projection_strain,
                             rigid_lattice_displacement, rotate_strain,
                             strain_from_deformation_gradient, tensor_strain)


def _graph(lat):
    return build_lattice_graph(lat.positions, lat.v1, lat.v2)


def _interior(lat, margin=20):
    return inside_image(lat.positions, lat.shape, margin)


# ============================================================ Method 1
def test_projection_strain_uniform():
    lat = synthetic.strained_lattice((240, 240), a=16.0, F=[[1.0, 0.0], [0.0, 1.02]])
    g = _graph(lat)
    along_y = projection_strain(g, lat.v2, d0_px=16.0, outlier_nsigma=None)
    along_x = projection_strain(g, lat.v1, d0_px=16.0, outlier_nsigma=None)
    np.testing.assert_allclose(along_y.strain[along_y.valid], 0.02, atol=1e-12)
    np.testing.assert_allclose(along_x.strain[along_x.valid], 0.0, atol=1e-12)
    assert along_y.valid.all()                   # border columns use one side
    np.testing.assert_allclose(along_y.spacing_px, 16.32)


def test_projection_strain_per_column_reference():
    lat = synthetic.superlattice((300, 200), a=16.0, c_values=(16.0, 17.0),
                                 cells_per_layer=5)
    g = _graph(lat)
    d0 = np.where(lat.labels % 2 == 0, 16.0, 17.0)          # each layer its own bulk
    m = projection_strain(g, lat.v2, d0_px=d0, outlier_nsigma=None)
    far_from_interfaces = np.isclose(lat.true_F[:, 1, 1] * 16.0, d0)
    np.testing.assert_allclose(m.strain[far_from_interfaces & m.valid], 0.0,
                               atol=1e-12)


def test_thin_strained_layer_is_not_an_outlier():
    """Regression: a global median/MAD test deletes a minority strained layer.

    One strained layer (c = 16.8 px) between thick unstrained ones: the global
    rule sees the whole layer as outliers, the local rule keeps it.
    """
    lat = synthetic.superlattice((400, 200), a=16.0, c_values=(16.0, 16.0, 16.0, 16.8),
                                 cells_per_layer=4, jitter=0.02, seed=0)
    g = _graph(lat)
    layer = np.isclose(lat.true_F[:, 1, 1], 1.05)
    assert 0 < layer.mean() < 0.3
    glob = projection_strain(g, lat.v2, 16.0, outlier_scope="global")
    loc = projection_strain(g, lat.v2, 16.0)                  # default: local
    assert np.all(glob.status[layer] == 4)                    # the pitfall
    assert np.all(loc.status[layer] == 0)
    np.testing.assert_allclose(loc.strain[layer], 0.05, atol=0.004)
    with pytest.raises(ValueError, match="outlier_scope"):
        projection_strain(g, lat.v2, 16.0, outlier_scope="row")


def test_projection_strain_status_codes():
    lat = synthetic.strained_lattice((200, 200), a=16.0)
    g = _graph(lat)
    pos = lat.positions.copy()
    # an isolated column far away has no neighbours at all
    pos = np.vstack([pos, [[1000.0, 1000.0]]])
    g2 = build_lattice_graph(pos, lat.v1, lat.v2)
    k = int(np.flatnonzero(inside_image(pos, (200, 200), 40))[0])
    g2.neighbor_vec[k, 2] += [0.0, 3.0]      # one over-long bond -> outlier
    m = projection_strain(g2, lat.v2, 16.0, image_shape=(200, 200), edge=20)
    assert m.status[-1] == 1                                   # no neighbour
    assert np.sum(m.status == 3) > 0                           # edge margin
    assert m.status[k] == 4                                    # robust outlier
    assert np.isnan(m.strain[m.status != 0]).all()
    summary = m.summary()
    assert sum(summary.values()) == len(pos)
    assert "Projection strain" in m.report()
    assert isinstance(m, ProjectionStrain)
    g.bad_edges[:] = True                                      # everything rejected
    rejected = projection_strain(g, lat.v2, 16.0)
    assert np.all(rejected.status == 2)


def test_projection_strain_validation():
    lat = synthetic.strained_lattice((120, 120), a=16.0)
    g = _graph(lat)
    with pytest.raises(ValueError, match="positive"):
        projection_strain(g, lat.v2, 0.0)
    with pytest.raises(ValueError, match="one value per column"):
        projection_strain(g, lat.v2, np.ones(3))
    with pytest.raises(TypeError):
        projection_strain(lat.positions, lat.v2, 16.0)


# ============================================================ Method 2
def test_fit_reference_lattice_recovers_vectors():
    lat = synthetic.strained_lattice((260, 260), a=16.0, c=17.0, rotation_deg=8.0)
    g = _graph(lat)
    ref = in_rectangle(lat.positions, (60, 200), (60, 200))
    fit = fit_reference_lattice(g, ref)
    assert isinstance(fit, ReferenceLattice)
    np.testing.assert_allclose(fit.a, lat.v1, atol=1e-9)
    np.testing.assert_allclose(fit.b, lat.v2, atol=1e-9)
    assert fit.reached.all()
    np.testing.assert_allclose(fit.ideal_positions(), lat.positions, atol=1e-9)
    # with position noise the worst 20 % of the reference columns are rejected
    noisy = synthetic.strained_lattice((260, 260), a=16.0, jitter=0.2, seed=2)
    gn = _graph(noisy)
    refn = in_rectangle(noisy.positions, (60, 200), (60, 200))
    fitn = fit_reference_lattice(gn, refn)
    assert fitn.used_mask.sum() == pytest.approx(0.8 * refn.sum(), abs=1)
    np.testing.assert_allclose(fitn.a, noisy.v1, atol=0.05)
    s = fit.summary()
    assert s["residual_max_px"] < 1e-9
    assert "Reference lattice" in fit.report(sampling=0.02)


def test_fit_reference_lattice_validation():
    lat = synthetic.strained_lattice((200, 200), a=16.0)
    g = _graph(lat)
    with pytest.raises(ValueError, match="columns in the reference region"):
        fit_reference_lattice(g, in_rectangle(lat.positions, (0, 20), (0, 20)))
    with pytest.raises(ValueError, match="one entry per column"):
        fit_reference_lattice(g, np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="reject_fraction"):
        fit_reference_lattice(g, np.ones(g.n_atoms, dtype=bool), reject_fraction=1)


def _lstsq_reference(graph, a, b):
    """Per-column np.linalg.lstsq, exactly as in the original notebook."""
    ref_vec = {0: a, 1: -a, 2: b, 3: -b}
    F = np.full((graph.n_atoms, 2, 2), np.nan)
    clean = graph.clean_idx
    for i in range(graph.n_atoms):
        E, M = [], []
        has0 = has1 = False
        for d in range(4):
            if clean[i, d] < 0:
                continue
            E.append(ref_vec[d])
            M.append(graph.neighbor_vec[i, d])
            has0 |= d in (0, 1)
            has1 |= d in (2, 3)
        if has0 and has1:
            FT, *_ = np.linalg.lstsq(np.array(E), np.array(M), rcond=None)
            F[i] = FT.T
    return F


def test_vectorised_gradient_equals_per_column_lstsq():
    lat = synthetic.strained_lattice((220, 220), a=16.0, jitter=0.6, seed=4)
    g = _graph(lat)
    g.bad_edges[::7, 1] = True                          # remove some bonds
    a, b = np.array([16.1, 0.2]), np.array([-0.1, 15.9])
    F, n_bonds = deformation_gradients(g, np.vstack([a, b]))
    F_ref = _lstsq_reference(g, a, b)
    np.testing.assert_array_equal(np.isnan(F), np.isnan(F_ref))
    ok = ~np.isnan(F[:, 0, 0])
    np.testing.assert_allclose(F[ok], F_ref[ok], atol=1e-12)
    assert n_bonds.max() == 4


@pytest.mark.parametrize("F_true", [
    [[1.01, 0.0], [0.0, 0.98]],                 # biaxial
    [[1.0, 0.015], [0.015, 1.0]],               # pure shear
    [[np.cos(0.02), -np.sin(0.02)], [np.sin(0.02), np.cos(0.02)]],   # rotation
    [[1.02, 0.01], [-0.005, 0.97]],             # general
])
def test_tensor_strain_homogeneous(F_true):
    F_true = np.array(F_true)
    lat = synthetic.strained_lattice((260, 260), a=16.0, F=F_true)
    g = build_lattice_graph(lat.positions, F_true @ lat.v1, F_true @ lat.v2)
    t = tensor_strain(g, np.vstack([lat.v1, lat.v2]))
    assert isinstance(t, TensorStrain)
    # every column with bonds along both axes has a tensor (all interior ones)
    assert t.valid[_interior(lat)].all() and t.valid.mean() > 0.97
    ok = t.valid
    np.testing.assert_allclose(t.F[ok], np.broadcast_to(F_true, t.F[ok].shape),
                               atol=1e-12)
    exx, eyy, exy, om = strain_from_deformation_gradient(F_true)
    np.testing.assert_allclose(t.exx[ok], exx, atol=1e-12)
    np.testing.assert_allclose(t.eyy[ok], eyy, atol=1e-12)
    np.testing.assert_allclose(t.exy[ok], exy, atol=1e-12)
    np.testing.assert_allclose(t.omega[ok], om, atol=1e-12)


def test_rotation_sign_convention():
    """Positive omega = rotation from +x towards +y (clockwise on screen)."""
    th = 0.03
    F = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    _, _, _, om = strain_from_deformation_gradient(F)
    assert om == pytest.approx(np.sin(th))


def test_strain_is_immune_to_reference_bias():
    """A biased reference gives a uniform offset, never a growing error.

    This is the property that motivated the per-column deformation gradient:
    the rigid-lattice displacement grows linearly with distance from the
    anchor, while the strain stays uniform.
    """
    lat = synthetic.strained_lattice((400, 400), a=16.0)
    g = _graph(lat)
    fit = fit_reference_lattice(g, in_rectangle(lat.positions, (170, 230), (170, 230)))
    biased = ReferenceLattice(fit.origin, 1.005 * fit.a, 1.005 * fit.b, fit.indices,
                              fit.reached, fit.anchor, fit.reference_mask,
                              fit.used_mask, fit.residuals)
    t = tensor_strain(g, biased)
    np.testing.assert_allclose(t.eyy, 1 / 1.005 - 1, atol=1e-12)
    np.testing.assert_allclose(t.exx, 1 / 1.005 - 1, atol=1e-12)
    assert np.nanstd(t.eyy) < 1e-12
    u = rigid_lattice_displacement(g, biased)
    dist = np.linalg.norm(g.positions[u.per_site["index"]] - g.positions[fit.anchor], axis=1)
    # u = -0.005 n.a exactly: proportional to the distance from the anchor
    np.testing.assert_allclose(u.magnitude, 0.005 * dist, atol=1e-9)
    assert u.magnitude.max() > 1.3                            # px, far away


def test_superlattice_relative_to_reference_layer():
    lat = synthetic.superlattice((360, 240), a=16.0, c_values=(16.0, 16.8),
                                 cells_per_layer=6)
    g = _graph(lat)
    fit = fit_reference_lattice(g, lat.labels == 0)
    t = tensor_strain(g, fit)
    expected = strain_from_deformation_gradient(lat.true_F)
    np.testing.assert_allclose(t.exx, expected[0], atol=1e-9)
    np.testing.assert_allclose(t.eyy, expected[1], atol=1e-9)
    np.testing.assert_allclose(t.exy, 0.0, atol=1e-9)
    inner = np.isclose(expected[1], 0.05)
    assert inner.sum() > 20                                    # layer interior
    np.testing.assert_allclose(t.eyy[inner], 0.05, atol=1e-9)


def test_per_column_reference_for_multilayers():
    """Each layer referred to its own bulk lattice: zero strain inside layers."""
    lat = synthetic.superlattice((360, 240), a=16.0, c_values=(16.0, 16.8),
                                 cells_per_layer=6)
    g = _graph(lat)
    c = np.where(lat.labels % 2 == 0, 16.0, 16.8)
    ref = np.zeros((g.n_atoms, 2, 2))
    ref[:, 0] = [16.0, 0.0]
    ref[:, 1, 1] = c
    t = tensor_strain(g, ref)
    interior = np.isclose(lat.true_F[:, 1, 1] * 16.0, c)
    np.testing.assert_allclose(t.eyy[interior], 0.0, atol=1e-9)
    with pytest.raises(ValueError, match="reference_vectors"):
        tensor_strain(g, np.ones((3, 2, 2)))


def test_noise_propagation_is_unbiased():
    lat = synthetic.strained_lattice((400, 400), a=16.0, F=[[1.0, 0.0], [0.0, 1.03]],
                                     jitter=0.15, seed=9)
    g = build_lattice_graph(lat.positions, lat.v1, [0, 16.48])
    t = tensor_strain(g, np.vstack([lat.v1, lat.v2]), mask=_interior(lat))
    assert np.nanmean(t.eyy) == pytest.approx(0.03, abs=1e-3)
    assert np.nanmean(t.exx) == pytest.approx(0.0, abs=1e-3)
    # with 0.15 px position noise and 16 px bonds, per-column std ~ 0.7 %
    assert 0.003 < np.nanstd(t.eyy) < 0.012


def test_mask_and_frames():
    lat = synthetic.strained_lattice((200, 200), a=16.0,
                                     F=[[1.01, 0.004], [0.004, 0.99]])
    g = build_lattice_graph(lat.positions, lat.v1, lat.v2)
    mask = _interior(lat)
    t = tensor_strain(g, np.vstack([lat.v1, lat.v2]), mask=mask)
    assert np.isnan(t.exx[~mask]).all() and np.isfinite(t.exx[mask]).all()
    assert t.n_bonds[~mask].max() == 0
    cart = t.in_frame("cartesian")
    np.testing.assert_allclose(cart.exy, -t.exy)
    np.testing.assert_allclose(cart.omega, -t.omega)
    np.testing.assert_allclose(cart.eyy, t.eyy)
    rot = t.in_frame("rotated", angle_deg=30)
    np.testing.assert_allclose(rot.exx + rot.eyy, t.exx + t.eyy,
                               atol=1e-12)                      # trace invariant
    with pytest.raises(ValueError):
        t.in_frame("rotated")
    with pytest.raises(ValueError):
        cart.in_frame("image")
    with pytest.raises(ValueError):
        t.component("ezz")
    assert "Tensor strain" in t.report()
    assert t.summary()["n_valid"] == mask.sum()


def test_rotate_strain_known_values():
    exx, eyy, exy = rotate_strain(0.01, 0.0, 0.0, np.pi / 2)
    assert (exx, eyy, exy) == pytest.approx((0.0, 0.01, 0.0), abs=1e-15)
    exx, eyy, exy = rotate_strain(0.01, -0.01, 0.0, np.pi / 4)
    assert (exx, eyy, exy) == pytest.approx((0.0, 0.0, -0.01), abs=1e-15)


# ============================================================ helpers
def test_interpolate_to_grid_linear_exact():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 100, (300, 2))
    vals = 0.3 * pts[:, 0] - 0.1 * pts[:, 1] + 2
    grid, xi, yi = interpolate_to_grid(pts, vals, step=5, extent=(20, 80, 20, 80))
    xx, yy = np.meshgrid(xi, yi)
    np.testing.assert_allclose(grid, 0.3 * xx - 0.1 * yy + 2, atol=1e-9)
    far, _, _ = interpolate_to_grid(pts, vals, step=10, extent=(-50, 150, -50, 150))
    assert np.isnan(far).any()
    with pytest.raises(ValueError):
        interpolate_to_grid(pts[:2], vals[:2])


def test_binned_profile_and_line_profile():
    lat = synthetic.superlattice((300, 160), a=16.0, c_values=(16.0, 16.8),
                                 cells_per_layer=5)
    eyy = lat.true_F[:, 1, 1] - 1
    prof = binned_profile(lat.positions, eyy, axis="y", bin_width=4.0)
    rows = prof["count"] > 0
    assert rows.sum() == len(np.unique(lat.positions[:, 1]))
    assert np.nanmax(prof["std"]) < 1e-12              # rows are uniform
    with pytest.raises(ValueError):
        binned_profile(lat.positions, eyy, axis="z")
    grid, xi, yi = interpolate_to_grid(lat.positions, eyy, step=2)
    coords, values = line_profile(grid, xi, yi, axis="x")
    assert len(coords) == len(values) == grid.shape[0]
    coords, values = line_profile(grid, xi, yi, axis="y", coordinate=100)
    assert len(values) == grid.shape[1]
    with pytest.raises(ValueError):
        line_profile(grid, xi, yi, axis="z")
