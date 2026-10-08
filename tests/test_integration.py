"""End-to-end tests: full pipelines on synthetic images and on the example data.

The example-data tests pin the numbers reported with the original notebooks
(e.g. 64 complete cages and a median Ti displacement of 20.545 pm for the
simulated PbTiO3 [100] image), so any change of behaviour is noticed.
"""

import importlib.util

import numpy as np
import pytest

import polarmap as pm
from polarmap import synthetic

HAS_HYPERSPY = importlib.util.find_spec("hyperspy") is not None
needs_hyperspy = pytest.mark.skipif(not HAS_HYPERSPY, reason="HyperSpy not installed")


def _data(data_dir, name):
    path = data_dir / name
    if not path.exists():
        pytest.skip(f"example file {name} not available")
    return path


# ------------------------------------------------------------------ synthetic
def test_displacement_pipeline_from_noisy_image():
    s = synthetic.perovskite_100(shape=(256, 256), a=25.0, displacement=(0.3, -1.2),
                                 dose=300, noise_sigma=0.01, seed=21, sampling=0.016)
    seeds = pm.detect_peaks(s.image, min_distance=22, threshold_rel=0.2,
                            exclude_border=5)
    A = pm.refine_gaussian(s.image, seeds, box=8)
    v1, v2 = pm.estimate_lattice_vectors(A)
    np.testing.assert_allclose([np.linalg.norm(v1), np.linalg.norm(v2)], 25.0, rtol=2e-3)
    field = pm.measure_cage_displacement(s.image, A, v1, v2)
    desc = pm.describe_displacements(field, s.sampling)
    true_pm = np.hypot(0.3, 1.2) * s.sampling * 1e3
    assert desc["median_pm"] == pytest.approx(true_pm, abs=0.6)
    true_angle = np.degrees(np.arctan2(1.2, 0.3))            # cartesian, y up
    assert desc["circular_mean_deg"] == pytest.approx(true_angle, abs=2.0)


def test_strain_pipeline_from_rendered_superlattice():
    lat = synthetic.superlattice((330, 260), a=16.0, c_values=(16.0, 16.8),
                                 cells_per_layer=6, render=True, sigma=2.4,
                                 dose=500, seed=5)
    seeds = pm.detect_peaks(lat.image, min_distance=13, threshold_rel=0.15,
                            exclude_border=3)
    pos, params = pm.refine_gaussian(lat.image, seeds, box=5, return_params=True)
    pos = pos[params["success"]]
    v1, v2 = pm.estimate_lattice_vectors(pos)
    graph = pm.build_lattice_graph(pos, v1, v2)
    assert graph.summary()["n_bonds_rejected"] == 0
    # reference: first layer (rows of layer 0 are at y < 6 * 16)
    ref = pm.in_rectangle(pos, (0, 260), (10, 80))
    fit = pm.fit_reference_lattice(graph, ref)
    t = pm.tensor_strain(graph, fit, mask=pm.inside_image(pos, lat.image.shape, 10))
    layer1 = pm.in_rectangle(pos, (20, 240), (120, 170))      # inside layer 1
    layer0 = pm.in_rectangle(pos, (20, 240), (20, 75))
    assert np.nanmedian(t.eyy[layer1]) == pytest.approx(0.05, abs=0.004)
    assert np.nanmedian(t.eyy[layer0]) == pytest.approx(0.0, abs=0.004)
    assert np.nanmedian(np.abs(t.exx)) < 0.01
    m1 = pm.projection_strain(graph, v2, d0_px=16.0, image_shape=lat.image.shape)
    assert np.nanmedian(m1.strain[layer1]) == pytest.approx(0.05, abs=0.004)


# ------------------------------------------------------------------ example data
@pytest.mark.hyperspy
@needs_hyperspy
def test_pto100_reproduces_published_numbers(data_dir):
    img, sampling = pm.load_image(_data(data_dir, "PTO10040nmHAADF.dm4"))
    seeds = pm.detect_peaks(img, min_distance=30, threshold_rel=0.20, exclude_border=5)
    A = pm.refine_gaussian(img, seeds, box=10)
    assert len(A) == 81
    precision = pm.compare_refinements(img, seeds, sampling, box=10)
    assert precision["rms_pm"] == pytest.approx(2.2, abs=0.1)
    v1, v2 = pm.estimate_lattice_vectors(A)
    np.testing.assert_allclose(v1, [24.4, 0.0], atol=0.01)
    np.testing.assert_allclose(v2, [0.0, 25.8], atol=0.01)
    field = pm.measure_cage_displacement(img, A, v1, v2, edge=9)
    assert len(field) == 64
    desc = pm.describe_displacements(field, sampling)
    assert desc["median_pm"] == pytest.approx(20.545, abs=0.005)
    assert desc["circular_mean_deg"] == pytest.approx(90.0, abs=0.2)   # pointing up


@pytest.mark.hyperspy
@needs_hyperspy
def test_pzo110_pair_model_is_antipolar(data_dir):
    img, sampling = pm.load_image(_data(data_dir, "PZO11040nmHAADF.dm4"))
    seeds = pm.detect_peaks(img, min_distance=8, threshold_rel=0.20)
    split = pm.split_by_amplitude(img, seeds, box=3, edge=10)
    assert (len(split.bright), len(split.dim)) == (135, 150)
    pv = pm.estimate_pair_vector(split.bright, axis_hint=(0, 1))
    np.testing.assert_allclose(pv, [0.0, 25.7], atol=0.05)
    field = pm.measure_pair_displacement(img, split.bright, pv)
    assert len(field) == 120
    ang = field.angle()
    right = np.abs(pm.angular_difference(ang, 0.0)) < 30
    left = np.abs(pm.angular_difference(ang, 180.0)) < 30
    assert right.sum() + left.sum() == len(field)             # all horizontal
    assert 0.3 < right.mean() < 0.7                           # antiparallel mix
    assert np.median(field.magnitude_in(sampling)) == pytest.approx(2.25, abs=0.1)


def test_superlattice_mrc_strain_profile(data_dir):
    """Method 2 on a crop of the experimental superlattice image."""
    with pytest.warns(UserWarning, match="extended header"):
        img, sampling = pm.load_image(_data(data_dir, "ImgOrigin.mrc"))
    crop = img[300:800, 0:400]                     # bright / dark / bright layers
    seeds = pm.detect_peaks(crop, min_distance=12, threshold_rel=0.1, exclude_border=4)
    pos, params = pm.refine_gaussian(crop, seeds, box=4, return_params=True)
    pos = pos[params["success"]]
    v1, v2 = pm.estimate_lattice_vectors(pos)
    assert np.linalg.norm(v1) * sampling == pytest.approx(0.406, abs=0.003)
    graph = pm.build_lattice_graph(pos, v1, v2)
    fit = pm.fit_reference_lattice(graph, pm.in_rectangle(pos, (20, 380), (150, 350)))
    t = pm.tensor_strain(graph, fit, mask=pm.inside_image(pos, crop.shape, 15))
    dark = pm.in_rectangle(pos, (20, 380), (160, 340))
    bright = pm.in_rectangle(pos, (20, 380), (15, 50))
    assert abs(np.nanmedian(t.eyy[dark])) < 0.003
    assert 0.035 < np.nanmedian(t.eyy[bright]) < 0.055
    assert abs(np.nanmedian(t.exx)) < 0.003                   # coherent in-plane
