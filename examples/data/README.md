# Example data

| File | Content | Calibration |
|---|---|---|
| `PTO10040nmHAADF.dm4` | Dr. Probe multislice simulation, PbTiO₃ [100], 40 nm thick, HAADF, probe-convolved | 0.016 nm/px (in file) |
| `PZO11040nmHAADF.dm4` | Dr. Probe multislice simulation, PbZrO₃ [110], 40 nm thick, HAADF, probe-convolved | 0.016 nm/px (in file) |
| `BCZT_STO_superlattice_strain.tif` | experimental HAADF-STEM image of an oxide superlattice  
The two simulated images are the sample images distributed with VecMap
(T. Ma, <https://github.com/matao1984/vec-map>, MIT License); please cite VecMap
when using them.
Reading the DM4 files requires HyperSpy (`pip install "polarmap[io]"`); the MRC
file is read by `polarmap` itself.
