"""Tests for polarmap.io: calibration handling and file readers."""

import importlib.util
import struct
import warnings

import numpy as np
import pytest

from polarmap import io
from polarmap.io import load_image, read_mrc, units_to_nm

HAS_HYPERSPY = importlib.util.find_spec("hyperspy") is not None


@pytest.mark.parametrize("unit, factor", [
    ("nm", 1.0), ("NM", 1.0), (" nm ", 1.0), ("Å", 0.1), ("A", 0.1),
    ("pm", 1e-3), ("µm", 1e3), ("um", 1e3), ("m", 1e9),
])
def test_units_to_nm_known(unit, factor):
    assert units_to_nm(unit) == pytest.approx(factor)


@pytest.mark.parametrize("unit", ["<undefined>", "px", "", None, "1/nm"])
def test_units_to_nm_unknown(unit):
    assert units_to_nm(unit) is None


def test_npy_requires_sampling(tmp_path):
    path = tmp_path / "img.npy"
    np.save(path, np.arange(12, dtype=np.uint16).reshape(3, 4))
    with pytest.raises(ValueError, match="no pixel calibration"):
        load_image(path)
    img, sampling = load_image(path, sampling=0.02)
    assert img.dtype == np.float64 and img.shape == (3, 4)
    assert sampling == 0.02
    assert img[2, 3] == 11


@pytest.mark.parametrize("bad", [0, -1, np.nan])
def test_invalid_sampling(tmp_path, bad):
    path = tmp_path / "img.npy"
    np.save(path, np.zeros((4, 4)))
    with pytest.raises(ValueError, match="positive"):
        load_image(path, sampling=bad)


def test_missing_file_and_bad_backend(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "nope.png", sampling=1)
    path = tmp_path / "img.npy"
    np.save(path, np.zeros((4, 4)))
    with pytest.raises(ValueError, match="backend"):
        load_image(path, sampling=1, backend="imagej")


def test_rgb_png_is_averaged(tmp_path):
    from PIL import Image
    rgb = np.zeros((5, 6, 3), dtype=np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 30, 60, 90
    path = tmp_path / "rgb.png"
    Image.fromarray(rgb).save(path)
    img, _ = load_image(path, sampling=0.01)
    assert img.shape == (5, 6)
    np.testing.assert_allclose(img, 60.0)          # plain channel average


def test_tiff_stack_frames(tmp_path):
    from PIL import Image
    frames = [np.full((4, 5), v, dtype=np.uint16) for v in (100, 200, 600)]
    path = tmp_path / "stack.tif"
    Image.fromarray(frames[0]).save(path, save_all=True,
                                    append_images=[Image.fromarray(f) for f in frames[1:]])
    mean, _ = load_image(path, sampling=0.01)
    np.testing.assert_allclose(mean, 300.0)
    second, _ = load_image(path, sampling=0.01, frames=1)
    np.testing.assert_allclose(second, 200.0)
    total, _ = load_image(path, sampling=0.01, frames="sum")
    np.testing.assert_allclose(total, 900.0)
    with pytest.raises(ValueError, match="frames"):
        load_image(path, sampling=0.01, frames="median")


def _write_mrc(path, data, cella_angstrom, nsymbt=0, include_ext=True, mode=2):
    """Write a minimal little-endian MRC file."""
    ny, nx = data.shape
    header = bytearray(1024)
    struct.pack_into("<4i", header, 0, nx, ny, 1, mode)
    struct.pack_into("<3i", header, 28, nx, ny, 1)
    struct.pack_into("<3f", header, 40, cella_angstrom, cella_angstrom * ny / nx, 1.0)
    struct.pack_into("<i", header, 92, nsymbt)
    header[104:108] = b"FEI2"
    header[208:212] = b"MAP "
    header[212:216] = bytes([0x44, 0x44, 0, 0])
    dtype = {2: "<f4", 6: "<u2", 1: "<i2"}[mode]
    with open(path, "wb") as fh:
        fh.write(header)
        if include_ext:
            fh.write(b"\x00" * nsymbt)
        fh.write(np.asarray(data, dtype=dtype).tobytes())


def test_read_mrc_roundtrip(tmp_path):
    data = np.arange(20, dtype=np.float32).reshape(4, 5)
    path = tmp_path / "a.mrc"
    _write_mrc(path, data, cella_angstrom=5 * 0.5)       # 0.5 Å/px = 0.05 nm/px
    arr, sampling, header = read_mrc(path)
    np.testing.assert_array_equal(arr, data)
    assert sampling == pytest.approx(0.05)
    assert header["nx"] == 5 and header["ny"] == 4 and header["exttyp"] == "FEI2"


def test_read_mrc_with_extended_header(tmp_path):
    data = np.arange(12, dtype=np.uint16).reshape(3, 4)
    path = tmp_path / "ext.mrc"
    _write_mrc(path, data, 4.0, nsymbt=128, include_ext=True, mode=6)
    arr, _, header = read_mrc(path)
    np.testing.assert_array_equal(arr, data)
    assert header["data_offset"] == 1024 + 128


def test_read_mrc_declared_but_missing_extended_header(tmp_path):
    """The situation of examples/data/ImgOrigin.mrc: NSYMBT > 0, block absent."""
    data = np.arange(12, dtype=np.uint16).reshape(3, 4)
    path = tmp_path / "stripped.mrc"
    _write_mrc(path, data, 4.0, nsymbt=909312, include_ext=False, mode=6)
    with pytest.warns(UserWarning, match="extended header"):
        arr, _, header = read_mrc(path)
    np.testing.assert_array_equal(arr, data)
    assert header["data_offset"] == 1024


def test_read_mrc_truncated(tmp_path):
    path = tmp_path / "trunc.mrc"
    _write_mrc(path, np.zeros((4, 4), np.float32), 4.0)
    raw = path.read_bytes()
    path.write_bytes(raw[:-10])
    with pytest.raises(ValueError, match="truncated"):
        read_mrc(path)


def test_load_image_mrc_calibration_and_override(tmp_path):
    path = tmp_path / "cal.mrc"
    _write_mrc(path, np.ones((4, 4), np.float32), cella_angstrom=4 * 0.2)
    _, sampling = load_image(path)
    assert sampling == pytest.approx(0.02)
    with pytest.warns(UserWarning, match="overriding"):
        _, sampling = load_image(path, sampling=0.03)
    assert sampling == 0.03
    with warnings.catch_warnings():
        warnings.simplefilter("error")              # a matching value is silent
        load_image(path, sampling=0.02)


def test_example_mrc(data_dir):
    path = data_dir / "ImgOrigin.mrc"
    if not path.exists():
        pytest.skip("example data not available")
    with pytest.warns(UserWarning, match="extended header"):
        img, sampling = load_image(path)
    assert img.shape == (1122, 1121)
    assert sampling == pytest.approx(0.03274, abs=1e-5)
    # The header statistics match the data read from byte 1024.
    assert img.min() == 12017 and img.max() == 33204
    assert img.mean() == pytest.approx(18486.19, abs=0.01)


@pytest.mark.hyperspy
@pytest.mark.skipif(not HAS_HYPERSPY, reason="HyperSpy not installed")
@pytest.mark.parametrize("name, shape", [("PTO10040nmHAADF.dm4", (258, 244)),
                                         ("PZO11040nmHAADF.dm4", (257, 294))])
def test_example_dm4(data_dir, name, shape):
    path = data_dir / name
    if not path.exists():
        pytest.skip("example data not available")
    img, sampling = load_image(path)
    assert img.shape == shape
    assert img.dtype == np.float64
    assert sampling == pytest.approx(0.016, rel=1e-6)


@pytest.mark.hyperspy
@pytest.mark.skipif(not HAS_HYPERSPY, reason="HyperSpy not installed")
def test_list_signals(data_dir):
    path = data_dir / "PTO10040nmHAADF.dm4"
    if not path.exists():
        pytest.skip("example data not available")
    titles = io.list_signals(path)
    assert len(titles) == 1 and "PTO100" in titles[0]


def test_select_signal_by_name_and_index():
    class Sig:
        def __init__(self, ndim):
            self.data = np.zeros((2,) * ndim)
    sigs = [Sig(1), Sig(2), Sig(2)]
    titles = ["spectrum", "HAADF", "iDPC"]
    assert io._select_signal(sigs, titles, None) is sigs[1]
    assert io._select_signal(sigs, titles, "idpc") is sigs[2]
    assert io._select_signal(sigs, titles, 0) is sigs[0]
    with pytest.raises(ValueError, match="no signal matching"):
        io._select_signal(sigs, titles, "DPC4")
