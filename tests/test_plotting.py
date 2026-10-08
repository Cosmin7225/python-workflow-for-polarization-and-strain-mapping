"""Smoke and behaviour tests for polarmap.plotting (Agg backend)."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from polarmap import plotting as pp
from polarmap import synthetic
from polarmap.displacement import measure_cage_displacement, measure_pair_displacement
from polarmap.lattice import build_lattice_graph
from polarmap.strain import (binned_profile, fit_reference_lattice,
                             projection_strain, tensor_strain)
from polarmap.validation import compare_displacement_fields


@pytest.fixture(scope="module")
def field(perov100):
    return measure_cage_displacement(perov100.image, perov100.reference,
                                     perov100.v1, perov100.v2)


@pytest.fixture(scope="module")
def strain_setup():
    lat = synthetic.superlattice((200, 160), a=16.0, c_values=(16.0, 16.8),
                                 cells_per_layer=4)
    g = build_lattice_graph(lat.positions, lat.v1, lat.v2)
    ref = fit_reference_lattice(g, lat.labels == 0)
    return lat, g, ref, tensor_strain(g, ref)


def test_nice_length_and_limits():
    assert pp.nice_length(7.3) == 5
    assert pp.nice_length(0.23) == pytest.approx(0.2)
    assert pp.nice_length(1.0) == 1
    with pytest.raises(ValueError):
        pp.nice_length(0)
    assert pp.symmetric_limits([-1, 2, np.nan], 100) == (-2.0, 2.0)
    assert pp.symmetric_limits([np.nan]) == (-1.0, 1.0)


@pytest.mark.parametrize("color_by", ["angle", "magnitude", None])
def test_vector_plots(perov100, field, color_by):
    fig, ax = pp.plot_displacement_vectors(field, perov100.image, color_by=color_by,
                                           sampling=0.016, scalebar_nm="auto",
                                           title="t")
    assert ax.get_title() == "t"
    assert ax.yaxis_inverted()


def test_vector_plot_errors(field):
    with pytest.raises(ValueError, match="color_by"):
        pp.plot_displacement_vectors(field, color_by="size")
    with pytest.raises(ValueError, match="sampling"):
        pp.plot_displacement_vectors(field, scalebar_nm=1.0)


@pytest.mark.parametrize("quantity", ["magnitude", "angle", "u", "v"])
@pytest.mark.parametrize("kind", ["tiles", "interpolated", "scatter"])
def test_displacement_maps(perov100, field, quantity, kind):
    fig, ax = pp.plot_displacement_map(field, quantity, image=perov100.image,
                                       sampling=0.016, kind=kind,
                                       colorbar_location="bottom")
    if quantity == "angle":                    # inset colour wheel, no colour bar
        assert len(fig.axes) == 1 and len(ax.child_axes) == 1
        assert ax.child_axes[0].name == "polar"
    else:                                      # map + appended colour bar
        assert len(fig.axes) == 2


def test_displacement_map_without_image(field):
    fig, ax = pp.plot_displacement_map(field, "magnitude", kind="tiles")
    x0, x1 = ax.get_xlim()
    assert x1 - x0 > 50 and ax.yaxis_inverted()
    with pytest.raises(ValueError):
        pp.plot_displacement_map(field, "speed")
    with pytest.raises(ValueError):
        pp.plot_displacement_map(field, kind="hexbin")


@pytest.mark.parametrize("convention, direction", [("cartesian", 1), ("image", -1)])
def test_colorwheel_matches_convention(convention, direction):
    fig, ax = plt.subplots()
    wax = pp.add_colorwheel(ax, convention=convention)
    assert wax.get_theta_direction() == direction
    with pytest.raises(ValueError):
        pp.add_colorwheel(ax, convention="polar")


@pytest.mark.parametrize("location", ["right", "left", "bottom", "top"])
def test_colorbar_locations(location):
    fig, ax = plt.subplots()
    im = ax.imshow(np.random.default_rng(0).random((10, 20)))
    cbar = pp.add_colorbar(im, ax, "label", location=location)
    expected = "horizontal" if location in ("bottom", "top") else "vertical"
    assert cbar.orientation == expected
    with pytest.raises(ValueError):
        pp.add_colorbar(im, ax, location="middle")


def test_scalebar_units():
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((100, 200)))
    bar = pp.add_scalebar(ax, sampling=0.02)
    assert "nm" in bar.txt_label.get_text()
    with pytest.raises(ValueError):
        pp.add_scalebar(ax, 0.02, data_units="mm")


def test_statistics_figure(field):
    fig, axes, desc = pp.plot_displacement_statistics(field, 0.016)
    assert len(axes) == 3 and desc["n"] == len(field)


def test_agreement_figure(field):
    cmp = compare_displacement_fields(field, field.with_sign(1), 0.016)
    fig, ax = pp.plot_magnitude_agreement(cmp)
    assert "bias" in ax.get_title()


def test_diagnostic_figures(perov100, perov110, field):
    pp.plot_columns(perov100.image, perov100.reference, label="A")
    pp.plot_cages(field, perov100.reference)
    pair = measure_pair_displacement(perov110.image, perov110.reference, perov110.v2)
    pp.plot_pairs(pair, perov110.reference)
    from polarmap.columns import detect_peaks, split_by_amplitude
    split = split_by_amplitude(perov110.image, detect_peaks(perov110.image, 8), edge=8)
    pp.plot_intensity_split(perov110.image, split)
    pp.plot_lattice_planes(perov100.image, perov100.reference, perov100.v1)
    pp.plot_neighbor_cloud(perov100.reference)


def test_strain_figures(strain_setup):
    lat, g, ref, t = strain_setup
    pp.plot_strain_scatter(lat.positions, t.eyy, sampling=0.02, scalebar_nm="auto")
    pp.plot_strain_triangulation(lat.positions, t.eyy, sampling=0.02)
    for kind in ("scatter", "grid"):
        fig, axes = pp.plot_strain_tensor(t, kind=kind, sampling=0.02)
        assert len(axes) == 4
    fig, axes = pp.plot_strain_tensor(t, components=("eyy",), omega_unit="mrad")
    assert len(axes) == 1
    m1 = projection_strain(g, lat.v2, 16.0)
    pp.plot_strain_comparison(lat.positions, t.eyy, m1.strain, kind="grid")
    prof = binned_profile(lat.positions, t.eyy, bin_width=4)
    fig, ax = pp.plot_profiles({"eyy": prof}, sampling=0.02)
    assert ax.get_xlabel() == "y (nm)"
    img = np.zeros(lat.shape)
    pp.plot_reference_region(img, g, ref, x_range=(0, 150), y_range=(0, 60), margin=10)
    pp.plot_neighbor_diagnostic(g, 5, image=img)
    with pytest.raises(ValueError):
        pp.plot_strain_tensor(t, kind="contour")
    with pytest.raises(ValueError):
        pp.plot_strain_triangulation(lat.positions[:2], t.eyy[:2])
