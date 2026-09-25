"""Tests for polarmap.columns: contrast handling, detection and refinement."""

import numpy as np
import pytest
from scipy.spatial import KDTree

from polarmap import columns, synthetic
from polarmap.columns import (as_bright_atoms, compare_refinements, detect_peaks,
                              flatten_background, refine_com, refine_gaussian,
                              remove_columns_gaussian, split_by_amplitude,
                              two_means_threshold)


def _match_error(found, truth):
    """Distance from each found point to the nearest true point."""
    d, _ = KDTree(truth).query(found)
    return d


# ---------------------------------------------------------------- contrast
def test_as_bright_atoms_modes():
    img = np.array([[0.0, 1.0], [2.0, 3.0]])
    adf = as_bright_atoms(img, "haadf")
    np.testing.assert_array_equal(adf, img)
    adf[0, 0] = 99                      # a copy, the input is untouched
    assert img[0, 0] == 0
    np.testing.assert_array_equal(as_bright_atoms(img, "ABF"), 3.0 - img)
    np.testing.assert_array_equal(as_bright_atoms(img, "iDPC"), img)


def test_as_bright_atoms_errors():
    with pytest.raises(ValueError, match="unknown image_mode"):
        as_bright_atoms(np.zeros((3, 3)), "SEM")
    with pytest.raises(ValueError, match="2-D"):
        as_bright_atoms(np.zeros((3, 3, 3)))


def test_flatten_background_range(perov100):
    out = flatten_background(perov100.image + np.linspace(0, 5, 210), sigma=40)
    assert out.min() == pytest.approx(0.0) and out.max() == pytest.approx(1.0)


# ---------------------------------------------------------------- detection
def test_detect_reference_sublattice_only(perov100):
    """With min_distance ~ A-A spacing only the bright A columns are found."""
    seeds = detect_peaks(perov100.image, min_distance=22, threshold_rel=0.2,
                         exclude_border=3)
    truth = perov100.reference
    inside = ((truth[:, 0] > 3) & (truth[:, 0] < 207)
              & (truth[:, 1] > 3) & (truth[:, 1] < 197))
    assert len(seeds) == inside.sum()
    assert _match_error(seeds, truth).max() <= 1.0


def test_detect_both_sublattices(perov100):
    seeds = detect_peaks(perov100.image, min_distance=12, threshold_rel=0.05,
                         exclude_border=3)
    both = np.vstack([perov100.reference, perov100.target])
    assert _match_error(seeds, both).max() <= 1.5
    assert len(seeds) > 1.5 * len(perov100.reference)


def test_detect_on_dark_contrast(perov100):
    abf = perov100.image.max() - perov100.image
    seeds_abf = detect_peaks(abf, 22, image_mode="ABF", threshold_rel=0.2)
    seeds_adf = detect_peaks(perov100.image, 22, threshold_rel=0.2)
    np.testing.assert_array_equal(seeds_abf, seeds_adf)


def test_tied_maxima_are_merged():
    img = np.zeros((20, 20))
    img[9:11, 9:11] = 1.0                    # a flat 2 x 2 plateau
    seeds = detect_peaks(img, 5, flatten=False)
    assert seeds.shape == (1, 2)
    np.testing.assert_allclose(seeds[0], [9.5, 9.5])


def test_detect_invalid_distance():
    with pytest.raises(ValueError):
        detect_peaks(np.zeros((5, 5)), 0)


# ---------------------------------------------------------------- refinement
def test_refine_gaussian_noise_free_accuracy(perov100):
    seeds = np.round(perov100.reference + 0.7)        # integer seeds, off by ~1 px
    refined, params = refine_gaussian(perov100.image, seeds, box=7,
                                      return_params=True)
    ok = params["success"]
    assert ok.sum() > 0.8 * len(seeds)
    err = np.linalg.norm(refined[ok] - perov100.reference[ok], axis=1)
    # Neighbour tails bias a single-Gaussian fit slightly; still << 0.05 px.
    assert err.max() < 0.05
    assert np.all(np.isnan(params["amplitude"][~ok]))
    np.testing.assert_array_equal(refined[~ok], seeds[~ok])   # seeds kept


def test_refine_gaussian_noisy_precision(perov100_noisy):
    truth = perov100_noisy.reference
    refined, params = refine_gaussian(perov100_noisy.image, np.round(truth),
                                      box=7, return_params=True)
    ok = params["success"]
    rms = np.sqrt(np.mean(np.sum((refined[ok] - truth[ok]) ** 2, axis=1)))
    assert rms < 0.1


def test_refine_gaussian_ellipticity():
    yy, xx = np.mgrid[0:31, 0:31]
    img = np.exp(-0.5 * (((xx - 15.3) / 3.0) ** 2 + ((yy - 14.6) / 1.5) ** 2))
    pos, params = refine_gaussian(img, [[15, 15]], box=8, return_params=True)
    np.testing.assert_allclose(pos[0], [15.3, 14.6], atol=1e-4)
    assert params["ellipticity"][0] == pytest.approx(2.0, rel=1e-3)


def test_refine_gaussian_edge_seed_is_flagged():
    img = np.zeros((20, 20))
    img[1, 1] = 1.0
    pos, params = refine_gaussian(img, [[1, 1]], box=4, return_params=True)
    assert not params["success"][0]
    np.testing.assert_array_equal(pos[0], [1, 1])


def test_refine_gaussian_empty_and_bad_shape():
    assert refine_gaussian(np.zeros((5, 5)), np.empty((0, 2))).shape == (0, 2)
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        refine_gaussian(np.zeros((5, 5)), [1, 2, 3])


def test_refine_com_reasonable(perov100):
    truth = perov100.reference
    refined = refine_com(perov100.image, np.round(truth), box=4)
    inside = (truth[:, 0] > 6) & (truth[:, 0] < 203) & (truth[:, 1] > 6) & (truth[:, 1] < 193)
    err = np.linalg.norm(refined[inside] - truth[inside], axis=1)
    assert err.max() < 0.35               # centre of mass is biased but close


def test_compare_refinements(perov100):
    out = compare_refinements(perov100.image, np.round(perov100.reference),
                              sampling=0.016, box=7)
    assert set(out) >= {"rms_pm", "median_pm", "n_compared", "gaussian", "com"}
    assert out["n_compared"] > 0 and np.isfinite(out["rms_pm"])


def test_remove_columns_gaussian(perov100):
    A = perov100.reference
    residual = remove_columns_gaussian(perov100.image, np.round(A), box=9)
    idx = np.round(A).astype(int)
    inside = (idx[:, 0] > 10) & (idx[:, 0] < 200) & (idx[:, 1] > 10) & (idx[:, 1] < 190)
    before = perov100.image[idx[inside, 1], idx[inside, 0]]
    after = residual[idx[inside, 1], idx[inside, 0]]
    assert np.all(after < 0.25 * before)
    # the B columns survive the subtraction
    b = np.round(perov100.target).astype(int)
    b_in = (b[:, 0] > 10) & (b[:, 0] < 200) & (b[:, 1] > 10) & (b[:, 1] < 190)
    assert np.median(residual[b[b_in, 1], b[b_in, 0]]) > 0.2


# ---------------------------------------------------------------- splitting
def test_two_means_threshold():
    values = np.r_[np.full(20, 1.0), np.full(30, 3.0)]
    t = two_means_threshold(values)
    assert 1.0 < t < 3.0
    with pytest.raises(ValueError):
        two_means_threshold([np.nan])


def test_split_by_amplitude(perov110):
    seeds = detect_peaks(perov110.image, 8, threshold_rel=0.2)
    split = split_by_amplitude(perov110.image, seeds, box=3, edge=8)
    assert _match_error(split.bright, perov110.reference).max() < 0.2
    assert _match_error(split.dim, perov110.target).max() < 0.3
    assert split.bright_amplitude.min() > split.threshold > split.dim_amplitude.max()


def test_split_needs_enough_columns():
    with pytest.raises(RuntimeError, match="Too few"):
        split_by_amplitude(np.zeros((30, 30)), [[15, 15]], box=3)


def test_public_mode_sets_are_disjoint():
    assert not (columns.BRIGHT_MODES & columns.DARK_MODES)
    assert synthetic is not None
