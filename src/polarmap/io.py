"""Loading calibrated atomic-resolution images.

Every analysis routine in :mod:`polarmap` works on a plain 2-D NumPy array and
a pixel size (``sampling``, in nm per pixel). This module turns image files into
that pair.

Supported inputs
----------------
* **Microscopy formats** (``.dm3``, ``.dm4``, ``.emd``, ``.hspy``, ``.hdf5``,
  ``.ser``, ...) are read through HyperSpy/RosettaSciIO, which preserves the
  pixel calibration stored in the file. HyperSpy is an *optional* dependency:
  install it with ``pip install "polarmap[io]"``.
* **MRC** files (``.mrc``), e.g. Thermo Fisher Velox exports, are read by a
  small built-in reader that needs no extra dependency and tolerates files whose
  header declares an extended header that is not actually present.
* **Display formats** (``.tif``, ``.png``, ``.jpg``, ...) and **NumPy** arrays
  (``.npy``) carry no reliable calibration; ``sampling`` must then be given
  explicitly. Colour images are converted to grey by averaging the RGB channels.

Units found in files (m, µm, nm, Å, pm) are converted to nanometres.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

__all__ = [
    "LENGTH_UNITS_TO_NM",
    "load_image",
    "read_mrc",
    "list_signals",
    "units_to_nm",
]

#: Conversion factors from common length units to nanometres.
LENGTH_UNITS_TO_NM = {
    "m": 1e9, "meter": 1e9, "metre": 1e9,
    "µm": 1e3, "μm": 1e3, "um": 1e3, "micron": 1e3,
    "micrometer": 1e3, "micrometre": 1e3,
    "nm": 1.0, "nanometer": 1.0, "nanometre": 1.0,
    "å": 0.1, "a": 0.1, "ang": 0.1, "angstrom": 0.1, "angstroem": 0.1,
    "pm": 1e-3, "picometer": 1e-3, "picometre": 1e-3,
}

_RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}
_NUMPY_SUFFIXES = {".npy"}
_MRC_SUFFIXES = {".mrc"}

# MRC data modes -> NumPy dtypes (MRC2014 specification).
_MRC_MODES = {0: np.int8, 1: np.int16, 2: np.float32, 6: np.uint16, 12: np.float16}


def units_to_nm(units):
    """Return the factor converting ``units`` to nanometres, or ``None``.

    Parameters
    ----------
    units : str
        Unit label as stored in a file, e.g. ``"nm"``, ``"Å"`` or ``"pm"``.
        Matching is case-insensitive and ignores surrounding whitespace.

    Returns
    -------
    float or None
        Multiplicative factor to nanometres, or ``None`` if the unit is not a
        recognised length unit (e.g. ``"<undefined>"`` or ``"px"``).
    """
    if units is None:
        return None
    return LENGTH_UNITS_TO_NM.get(str(units).strip().lower())


def load_image(path, sampling=None, *, signal=None, frames="mean",
               backend="auto"):
    """Load a 2-D image and its pixel size.

    Parameters
    ----------
    path : str or path-like
        Image file.
    sampling : float, optional
        Pixel size in **nm per pixel**. Required for files that carry no
        calibration (PNG/JPG/TIFF/NPY). For calibrated files it *overrides* the
        stored calibration; a warning is issued if the two differ by more than
        1 %.
    signal : int or str, optional
        For files holding several signals (e.g. a Velox ``.emd`` with HAADF,
        iDPC, DPC, ...): the index, or a case-insensitive substring of the
        signal title (e.g. ``"HAADF"``). By default the first 2-D signal is
        used. See :func:`list_signals`.
    frames : {"mean", "sum"} or int, default "mean"
        How a stack of frames is reduced to one image.
    backend : {"auto", "hyperspy"}, default "auto"
        ``"auto"`` reads MRC, raster and NumPy files with built-in readers and
        everything else with HyperSpy; ``"hyperspy"`` forces HyperSpy for all
        formats (useful for TIFF files exported with an embedded calibration).

    Returns
    -------
    image : numpy.ndarray
        2-D ``float64`` array (rows = y, columns = x).
    sampling : float
        Pixel size in nm/px.

    Raises
    ------
    ValueError
        If no calibration is available and ``sampling`` was not given, or the
        data cannot be reduced to a single 2-D image.
    ImportError
        If HyperSpy is needed but not installed.

    Examples
    --------
    >>> image, sampling = load_image("scan.dm4")              # doctest: +SKIP
    >>> image, sampling = load_image("scan.png", sampling=0.016)  # doctest: +SKIP
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()

    if backend not in ("auto", "hyperspy"):
        raise ValueError("backend must be 'auto' or 'hyperspy'")

    if backend == "hyperspy" or suffix not in (
            _RASTER_SUFFIXES | _NUMPY_SUFFIXES | _MRC_SUFFIXES):
        data, file_sampling, pixel_aspect = _read_with_hyperspy(path, signal)
    elif suffix in _MRC_SUFFIXES:
        # Re-issue reader warnings so that they point to the caller's code.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            data, file_sampling, _ = read_mrc(path)
        for w in caught:
            warnings.warn(w.message, w.category, stacklevel=2)
        pixel_aspect = 1.0
    elif suffix in _NUMPY_SUFFIXES:
        data, file_sampling, pixel_aspect = np.load(path), None, 1.0
    else:
        data, file_sampling, pixel_aspect = _read_raster(path), None, 1.0

    image = _to_single_image(np.asarray(data), frames)
    final_sampling = _resolve_sampling(file_sampling, sampling, path)
    if pixel_aspect is not None and abs(pixel_aspect - 1.0) > 0.01:
        warnings.warn(
            f"{path.name}: x and y pixel sizes differ by "
            f"{100 * abs(pixel_aspect - 1):.1f} %; polarmap assumes square "
            "pixels, so distances along y will be biased.", stacklevel=2)
    return image, final_sampling


def list_signals(path):
    """Return the titles of all signals stored in a (multi-signal) file.

    Useful for Velox ``.emd`` files before calling
    ``load_image(path, signal=...)``. Requires HyperSpy.
    """
    hs = _import_hyperspy()
    loaded = hs.load(str(path))
    signals = loaded if isinstance(loaded, (list, tuple)) else [loaded]
    return [_signal_title(s, i) for i, s in enumerate(signals)]


def read_mrc(path):
    """Read an MRC/CCP4 image file without external dependencies.

    Thermo Fisher (FEI) exports declare an extended header of ``NSYMBT`` bytes
    after the 1024-byte main header. Some processing tools strip that block
    without updating ``NSYMBT``, which makes other readers seek past the end of
    the file. When the file size shows that the data start directly after the
    main header, this reader does so and issues a warning.

    Parameters
    ----------
    path : str or path-like

    Returns
    -------
    data : numpy.ndarray
        Array of shape ``(ny, nx)`` or ``(nz, ny, nx)`` for stacks.
    sampling : float or None
        Pixel size in nm/px from the cell dimensions (``CELLA / MX``, stored in
        ångström), or ``None`` if the header carries no cell information.
    header : dict
        Selected header fields (``nx, ny, nz, mode, nsymbt, exttyp, ...``).
    """
    path = Path(path)
    file_size = os.path.getsize(path)
    with open(path, "rb") as fh:
        raw = fh.read(1024)
    if len(raw) < 1024:
        raise ValueError(f"{path.name}: file too short to be an MRC file")

    # Machine stamp (bytes 212-215): 0x11 0x11 -> big endian, else little.
    byteorder = ">" if raw[212] == 0x11 else "<"
    ints = np.frombuffer(raw, dtype=byteorder + "i4")
    floats = np.frombuffer(raw, dtype=byteorder + "f4")
    nx, ny, nz, mode = (int(v) for v in ints[0:4])
    if not (0 < nx < 1_000_000 and 0 < ny < 1_000_000 and 0 < nz < 1_000_000):
        raise ValueError(f"{path.name}: implausible MRC dimensions {nx, ny, nz}")
    if mode not in _MRC_MODES:
        raise ValueError(f"{path.name}: unsupported MRC data mode {mode}")

    dtype = np.dtype(_MRC_MODES[mode]).newbyteorder(byteorder)
    nsymbt = int(ints[23])
    n_values = nx * ny * nz
    n_bytes = n_values * dtype.itemsize
    offset = 1024 + nsymbt
    if offset + n_bytes > file_size:
        if 1024 + n_bytes == file_size:
            warnings.warn(
                f"{path.name}: header declares a {nsymbt}-byte extended header "
                "that is not present in the file; reading the image data "
                "directly after the 1024-byte main header.", stacklevel=2)
            offset = 1024
        else:
            raise ValueError(
                f"{path.name}: file is truncated ({file_size} bytes, expected "
                f"at least {offset + n_bytes})")

    data = np.fromfile(path, dtype=dtype, count=n_values, offset=offset)
    data = data.reshape(nz, ny, nx)
    if nz == 1:
        data = data[0]

    mx = int(ints[7])
    cella_x = float(floats[10])
    sampling = cella_x / mx / 10.0 if (mx > 0 and cella_x > 0) else None

    header = dict(nx=nx, ny=ny, nz=nz, mode=mode, nsymbt=nsymbt,
                  exttyp=raw[104:108].decode("latin-1"),
                  cella_angstrom=tuple(float(v) for v in floats[10:13]),
                  mxyz=tuple(int(v) for v in ints[7:10]),
                  data_offset=offset)
    return data, sampling, header


# ----------------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------------
def _import_hyperspy():
    try:
        import hyperspy.api as hs
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Reading this file format requires HyperSpy. Install it with\n"
            '    pip install "polarmap[io]"\n'
            "or convert the image to .mrc/.tif/.npy and pass `sampling`."
        ) from exc
    return hs


def _signal_title(sig, index):
    try:
        title = str(sig.metadata.General.title)
    except AttributeError:
        title = ""
    return title if title else f"signal {index}"


def _select_signal(signals, titles, signal):
    if signal is None:
        for s in signals:                       # prefer a 2-D image signal
            if np.asarray(s.data).ndim >= 2:
                return s
        return signals[0]
    if isinstance(signal, (int, np.integer)):
        return signals[int(signal)]
    key = str(signal).lower()
    for s, title in zip(signals, titles, strict=True):
        if key in title.lower():
            return s
    raise ValueError(f"no signal matching {signal!r}; available: {titles}")


def _read_with_hyperspy(path, signal):
    """Return (data, sampling_nm_or_None, pixel_aspect) read by HyperSpy."""
    hs = _import_hyperspy()
    loaded = hs.load(str(path))
    if isinstance(loaded, (list, tuple)):
        signals = list(loaded)
        titles = [_signal_title(s, i) for i, s in enumerate(signals)]
        sig = _select_signal(signals, titles, signal)
    else:
        sig = loaded

    data = np.asarray(sig.data)
    if data.dtype.names:                        # HyperSpy RGB(A) structured dtype
        data = np.mean([data[name].astype(float)
                        for name in data.dtype.names[:3]], axis=0)

    try:
        axes = list(sig.axes_manager.signal_axes)
    except AttributeError:
        axes = []
    sampling, aspect = None, 1.0
    if len(axes) >= 2:
        fx = units_to_nm(axes[0].units)
        fy = units_to_nm(axes[1].units)
        sx, sy = float(axes[0].scale), float(axes[1].scale)
        if fx is not None and sx > 0:
            sampling = sx * fx
            if fy is not None and sy > 0:
                aspect = (sy * fy) / sampling
    return data, sampling, aspect


def _read_raster(path):
    from PIL import Image, ImageSequence   # Pillow is a Matplotlib dependency

    with Image.open(path) as im:
        pages = [_page_to_array(page) for page in ImageSequence.Iterator(im)]
    return pages[0] if len(pages) == 1 else np.stack(pages)


def _page_to_array(page):
    if page.mode in ("RGB", "RGBA", "P", "LA", "CMYK"):
        rgb = np.asarray(page.convert("RGB"), dtype=float)
        return rgb.mean(axis=-1)                # plain channel average
    if page.mode == "I;16B":
        page = page.convert("I")
    return np.asarray(page)


def _to_single_image(data, frames):
    if data.ndim == 3 and data.shape[-1] in (3, 4) and data.shape[0] > 4:
        data = data[..., :3].astype(float).mean(axis=-1)    # (H, W, RGB[A])
    elif data.ndim == 3:
        data = _reduce_frames(data, frames)
    if data.ndim != 2:
        raise ValueError(f"cannot reduce data of shape {data.shape} to a 2-D image")
    return np.asarray(data, dtype=np.float64)


def _reduce_frames(data, frames):
    if isinstance(frames, str) and frames == "mean":
        return data.astype(float).mean(axis=0)
    if isinstance(frames, str) and frames == "sum":
        return data.astype(float).sum(axis=0)
    if isinstance(frames, (int, np.integer)):
        return data[int(frames)]
    raise ValueError("frames must be 'mean', 'sum' or an integer frame index")


def _resolve_sampling(file_sampling, user_sampling, path):
    if user_sampling is not None:
        user_sampling = float(user_sampling)
        if not np.isfinite(user_sampling) or user_sampling <= 0:
            raise ValueError("sampling must be a positive number (nm/px)")
        if (file_sampling is not None
                and abs(file_sampling - user_sampling) > 0.01 * user_sampling):
            warnings.warn(
                f"{path.name}: overriding the file calibration "
                f"({file_sampling:.6g} nm/px) with sampling={user_sampling:.6g} "
                "nm/px.", stacklevel=3)
        return user_sampling
    if file_sampling is None:
        raise ValueError(
            f"{path.name} carries no pixel calibration. Pass sampling=<nm per "
            "pixel>, or load the original calibrated file (dm3/dm4/emd/mrc).")
    return float(file_sampling)
