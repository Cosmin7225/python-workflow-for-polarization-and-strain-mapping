# Example data

| File | Content | Calibration |
|---|---|---|
| `PTO10040nmHAADF.dm4` | Dr. Probe multislice simulation, PbTiO₃ [100], 40 nm thick, HAADF, probe-convolved | 0.016 nm/px (in file) |
| `PZO11040nmHAADF.dm4` | Dr. Probe multislice simulation, PbZrO₃ [110], 40 nm thick, HAADF, probe-convolved | 0.016 nm/px (in file) |
| `ImgOrigin.mrc` | experimental HAADF-STEM image of an oxide superlattice (Thermo Fisher MRC export) | 0.03274 nm/px (in header) |

The two simulated images are the sample images distributed with VecMap
(T. Ma, <https://github.com/matao1984/vec-map>, MIT License); please cite VecMap
when using them.

`ImgOrigin.mrc` was previously stored as `ImgOrigin.dm4`, but it is an MRC
file. Its header declares a 909 312-byte Thermo Fisher extended header that is
not present in the file (the file size equals the 1024-byte main header plus
the 1121 × 1122 × 2-byte image, and the image statistics match the header's
min/max/mean). `polarmap.load_image` reads it correctly and warns about the
inconsistency; other readers may fail on it.

Reading the DM4 files requires HyperSpy (`pip install "polarmap[io]"`); the MRC
file is read by `polarmap` itself.
