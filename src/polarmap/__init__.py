"""polarmap: atomic-column displacement and strain mapping from STEM images.

A reproducible, tested implementation of the workflow for ferroelectric
perovskites (and other polar structures) described in the accompanying
manuscript. The package is organised by task:

=============================  ===============================================
Module                         Purpose
=============================  ===============================================
:mod:`polarmap.io`             load calibrated images (DM3/DM4, EMD, MRC, TIFF...)
:mod:`polarmap.columns`        column detection and sub-pixel refinement
:mod:`polarmap.lattice`        lattice vectors and the validated lattice graph
:mod:`polarmap.displacement`   displacement maps (cage, pair and offset models)
:mod:`polarmap.strain`         strain: Method 1 (projection) and Method 2 (tensor)
:mod:`polarmap.statistics`     angle conventions, circular and robust statistics
:mod:`polarmap.validation`     comparison with VecMap, GPA, ground truth
:mod:`polarmap.synthetic`      synthetic images with known ground truth
:mod:`polarmap.plotting`       publication figures (imported on first use)
=============================  ===============================================

The most common functions are also available at the top level, e.g.
``polarmap.load_image`` or ``polarmap.measure_cage_displacement``.
"""

from ._version import __version__
from .columns import (as_bright_atoms, compare_refinements, detect_peaks,
                      flatten_background, refine_com, refine_gaussian,
                      remove_columns_gaussian, split_by_amplitude)
from .displacement import (DisplacementField, displacement_from_cage,
                           displacement_from_offset,
                           displacement_from_sublattices,
                           displacement_to_polarization,
                           find_complete_perovskite_cages,
                           find_complete_reference_pairs,
                           measure_cage_displacement, measure_pair_displacement)
from .geometry import in_rectangle, inside_image
from .io import list_signals, load_image, read_mrc
from .lattice import (LatticeGraph, bfs_lattice_indices, build_lattice_graph,
                      estimate_lattice_vectors, estimate_pair_vector,
                      find_bad_edges_by_closure)
from .statistics import (angular_difference, circular_statistics,
                         describe_displacements, displacement_angle,
                         format_descriptors, local_outlier_mask,
                         mad_outlier_mask)
from .strain import (ProjectionStrain, ReferenceLattice, TensorStrain,
                     binned_profile, fit_reference_lattice, interpolate_to_grid,
                     line_profile,
                     projection_strain, rigid_lattice_displacement,
                     rotate_strain, tensor_strain)
from .validation import (compare_displacement_fields, compare_maps,
                         load_vecmap_csv, mutual_nearest_matches)

__all__ = [
    "__version__",
    # io
    "load_image", "read_mrc", "list_signals",
    # columns
    "as_bright_atoms", "flatten_background", "detect_peaks", "refine_gaussian",
    "refine_com", "compare_refinements", "remove_columns_gaussian",
    "split_by_amplitude",
    # geometry
    "inside_image", "in_rectangle",
    # lattice
    "estimate_lattice_vectors", "estimate_pair_vector", "LatticeGraph",
    "build_lattice_graph", "find_bad_edges_by_closure", "bfs_lattice_indices",
    # displacement
    "DisplacementField", "find_complete_perovskite_cages",
    "measure_cage_displacement", "find_complete_reference_pairs",
    "measure_pair_displacement", "displacement_from_sublattices",
    "displacement_from_cage", "displacement_from_offset",
    "displacement_to_polarization",
    # strain
    "ProjectionStrain", "projection_strain", "ReferenceLattice",
    "fit_reference_lattice", "rigid_lattice_displacement", "TensorStrain",
    "tensor_strain", "rotate_strain", "interpolate_to_grid", "binned_profile",
    "line_profile",
    # statistics
    "displacement_angle", "angular_difference", "circular_statistics",
    "mad_outlier_mask", "local_outlier_mask", "describe_displacements",
    "format_descriptors",
    # validation
    "load_vecmap_csv", "mutual_nearest_matches", "compare_displacement_fields",
    "compare_maps",
]


def __getattr__(name):
    # Import the plotting (Matplotlib) and synthetic subpackages on first use
    # so that `import polarmap` stays light and works on headless machines.
    if name in ("plotting", "synthetic"):
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module 'polarmap' has no attribute {name!r}")
