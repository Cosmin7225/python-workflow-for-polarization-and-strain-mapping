# Examples

The notebooks in the `examples/` folder of the repository run the complete
workflows on the example data in `examples/data/`. They are shown here with
their outputs; to run them yourself, install the package with the `all` extra
and start JupyterLab from the repository root.

| Notebook | Data | Shows |
|---|---|---|
| [01 Quick start (synthetic)](examples/01_quickstart_synthetic.ipynb) | generated | recovering a known displacement; no files or HyperSpy needed |
| [02 Perovskite [100] cages](examples/02_displacement_perovskite_100.ipynb) | `PTO10040nmHAADF.dm4` | the complete-cage workflow, statistics and all figure types |
| [03 Perovskite [110] pairs](examples/03_displacement_perovskite_110.ipynb) | `PZO11040nmHAADF.dm4` | intensity split, complete pairs, antipolar displacements |
| [04 Strain across a superlattice](examples/04_strain_superlattice.ipynb) | `ImgOrigin.mrc` | lattice graph, Methods 1 and 2, layer profiles, interfaces |

The two DM4 files are Dr. Probe multislice simulations of PbTiO₃ [100] and
PbZrO₃ [110] (40 nm thick, HAADF, probe-convolved) from the VecMap repository;
`ImgOrigin.mrc` is an experimental HAADF-STEM image of an oxide superlattice.

```{toctree}
:hidden:

examples/01_quickstart_synthetic
examples/02_displacement_perovskite_100
examples/03_displacement_perovskite_110
examples/04_strain_superlattice
```
