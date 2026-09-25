# Algorithm overview

The two workflows share the image handling and column localisation, then
branch into displacement and strain analysis. Function names are given in each
box; every step is documented in the user guide.

## Displacement workflow

```mermaid
flowchart TD
    A["Calibrated image<br/>load_image"] --> B["Contrast: bright columns<br/>image_mode (ADF / iDPC / ABF)"]
    B --> C["Seeds: local maxima<br/>detect_peaks (min_distance = reference spacing)"]
    C --> D["Sub-pixel positions: 2-D Gaussian fit<br/>refine_gaussian"]
    D --> E["Lattice vectors v1, v2<br/>estimate_lattice_vectors"]
    E --> F{"Projection"}
    F -->|"perovskite [100]"| G["Complete four-corner cages<br/>reference = centroid of 4 measured A columns"]
    F -->|"perovskite [110]"| H["Complete pairs<br/>reference = (1-f) r0 + f r1"]
    F -->|"dumbbell / wurtzite"| I["Fixed basis offset<br/>displacement_from_offset"]
    G --> J["Subtract reference columns<br/>remove_columns_gaussian"]
    H --> J
    J --> K["Fit target column on residual<br/>refine_gaussian"]
    K --> L["Quality control<br/>fit success, max displacement, amplitude"]
    I --> M
    L --> M["DisplacementField (u, v) = target - reference"]
    M --> N["Statistics: MAD outlier rule,<br/>circular statistics"]
    M --> O["Figures: arrows, colour maps,<br/>histograms, rose"]
    M --> P["Export: CSV; optional Born-charge estimate"]
```

## Strain workflow

```mermaid
flowchart TD
    A["Refined column positions<br/>(same steps as above)"] --> B["Lattice vectors v1, v2"]
    B --> C["Lattice graph: neighbour along +-v1, +-v2<br/>smallest perpendicular offset, within 0.5 |v|"]
    C --> D["Loop-closure test on every elementary cell<br/>reject inconsistent bonds"]
    D --> E{"Method"}
    E -->|"1: external reference d0"| F["Spacing along u from validated bonds<br/>symmetric average of + and - bonds"]
    F --> G["Strain (d - d0) / d0<br/>edge margin, local outlier test"]
    E -->|"2: internal reference region"| H["BFS integer indices from an anchor<br/>through validated bonds"]
    H --> I["Least-squares fit origin, a, b in the region<br/>reject worst 20 %, refit"]
    I --> J["Per-column deformation gradient F<br/>least squares m_d = F e_d over own bonds"]
    J --> K["Strain = sym(F) - I, rotation = asym(F)"]
    G --> L["Per-column maps, triangulation,<br/>binned layer profiles"]
    K --> L
    K --> M["Frames: image, Cartesian, rotated"]
```

## Why these choices?

| Step | Choice | Failure mode it avoids |
|---|---|---|
| Displacement reference | centroid of the *measured* complete cage or pair | a global lattice extrapolated over the image drifts away from the local lattice; incomplete cells produce false references |
| Target fit | on the image with reference columns subtracted | tails of bright columns pull the fit of the weak column |
| Neighbour matching | smallest perpendicular offset, limited to half a spacing | picking a column of the adjacent row under shear, or a diagonal column when the neighbour is missing |
| Bond validation | loop closure on each cell | a wrong bond silently propagating into the indexing or the strain |
| Indexing | breadth-first search through validated bonds | "phase slips" when rounding distances from a distant origin |
| Strain (Method 2) | per-column deformation gradient | linear drift of rigid-lattice displacements; noise amplification by differentiating interpolated fields |
| Outliers | median/MAD rules, local for maps | percentile clipping always removes data; global rules delete minority layers |
