# Loading images and calibration

Every analysis function works on a 2-D NumPy array and, where physical units
are involved, the pixel size `sampling` in **nm per pixel**.
{func}`polarmap.io.load_image` returns both:

```python
image, sampling = pm.load_image("scan.dm4")
```

## Formats and where the calibration comes from

| Format | Reader | Calibration |
|---|---|---|
| `.dm3`, `.dm4`, `.emd` (Velox), `.hspy`, `.hdf5`, `.ser`, ... | HyperSpy (optional extra `io`) | read from the file |
| `.mrc` | built-in | from the header cell size (`CELLA / MX`, ångström) |
| `.tif`, `.png`, `.jpg`, `.bmp` | Pillow (bundled with Matplotlib) | **none**: pass `sampling=` |
| `.npy` | NumPy | **none**: pass `sampling=` |

Units stored in files (m, µm, nm, Å, pm) are converted to nm. If `sampling` is
given for a calibrated file, it overrides the file value and a warning is
issued when the two differ by more than 1 %.

```{warning}
Display formats (PNG, JPEG, TIFF exported for presentation) do not carry a
reliable calibration. Load the original DM/EMD/MRC file whenever possible. If you
must use a display format, measure the pixel size on a known lattice spacing
first. Strain from **Method 1** is directly proportional to any calibration
error (a 1 % error in `sampling` shifts every strain value by about 1 %);
**Method 2** is independent of the calibration.
```

HyperSpy's TIFF reader understands calibrations written by DigitalMicrograph
and ImageJ; use it with `pm.load_image("img.tif", backend="hyperspy")`.

## Multi-detector files and frame stacks

Velox `.emd` files often contain several signals (HAADF, iDPC, DPC segments).
List them and select one by index or by a part of its title:

```python
pm.list_signals("scan.emd")            # ['HAADF', 'iDPC', 'DPC A-C', ...]
image, sampling = pm.load_image("scan.emd", signal="HAADF")
```

A stack of frames is reduced to one image with `frames="mean"` (default),
`frames="sum"` or a frame index, e.g. `frames=0`. For drift-corrupted series,
register the frames first (e.g. non-rigid registration) and load the result.

## Colour images

RGB(A) images are converted to grey by averaging the three colour channels
(not with luminance weights, so that equal-intensity exports are unchanged).

## MRC files from Thermo Fisher software

Thermo Fisher MRC files declare an extended header (`NSYMBT` bytes) between the
1024-byte main header and the data. Some tools remove this block without
updating `NSYMBT`, so other readers fail with "file too short" or "mmap length
is greater than file size". {func}`polarmap.io.read_mrc` detects this case from the
file size, reads the data directly after the main header and issues a warning.
The example `examples/data/ImgOrigin.mrc` is such a file.

## Contrast: which columns are bright?

All detection and fitting routines expect atomic columns to be *bright*. Pass
`image_mode` to the functions of {mod}`polarmap.columns` and
{mod}`polarmap.displacement`:

| `image_mode` | Treatment |
|---|---|
| `"ADF"`, `"HAADF"`, `"LAADF"`, `"MAADF"`, `"DF"` | used as is |
| `"iDPC"`, `"DPC"` | used as is (light and heavy columns have comparable intensity) |
| `"ABF"`, `"BF"` | inverted (`max - image`), because columns are dark |

This only fixes the sign of the contrast. It does not correct its physics: the
apparent position of a column can depend on the imaging mode, defocus, residual
aberrations, specimen tilt and thickness, particularly for light columns. The
choice of signal is therefore part of the experimental design; see
[Limitations](../limitations.md).
