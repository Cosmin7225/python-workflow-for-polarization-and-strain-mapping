"""Shared fixtures for the polarmap test suite.

Most tests run on synthetic data with exactly known ground truth
(:mod:`polarmap.synthetic`), so they are fast and need no data files. Tests on
the example images in ``examples/data`` are skipped when the files (or, for
the DM4 files, HyperSpy) are not available.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")                    # never open windows during tests

from polarmap import synthetic  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture(scope="session")
def data_dir():
    return DATA_DIR


@pytest.fixture(scope="session")
def perov100():
    """Noise-free perovskite [100] image with a uniform B displacement."""
    return synthetic.perovskite_100(shape=(200, 210), a=25.0,
                                    displacement=(0.4, -1.3))


@pytest.fixture(scope="session")
def perov100_noisy():
    """Same geometry with shot and read noise (fixed seed)."""
    return synthetic.perovskite_100(shape=(200, 210), a=25.0,
                                    displacement=(0.4, -1.3), dose=400,
                                    noise_sigma=0.01, seed=7)


@pytest.fixture(scope="session")
def perov110():
    """Two-column (pair) reference model with a uniform target displacement."""
    return synthetic.perovskite_110(shape=(210, 220), displacement=(0.6, -0.8))


@pytest.fixture
def square_points():
    """A perfect 12 x 10 square lattice (spacing 10 px) and its indices."""
    i, j = np.meshgrid(np.arange(12), np.arange(10))
    idx = np.column_stack([i.ravel(), j.ravel()])
    return 20.0 + 10.0 * idx.astype(float), idx
