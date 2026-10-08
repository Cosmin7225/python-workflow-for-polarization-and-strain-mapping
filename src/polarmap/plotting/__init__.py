"""Publication-quality figures (Matplotlib).

Every function draws into a Matplotlib ``Axes`` (a new figure is created when
``ax``/``axes`` is not given) and returns the figure and axes, so panels can be
combined freely and saved with ``fig.savefig(...)``. Nothing is shown or saved
automatically.

Conventions shared by all functions:

* images are shown in pixel coordinates with ``y`` pointing down;
* ``scalebar_nm`` adds a scale bar (``"auto"`` picks a round length);
* ``colorbar_location`` places colour bars at ``"right"`` or ``"bottom"``;
  colour bars always match the size of the image axes;
* orientations are colour coded with a cyclic colour map and a colour wheel
  drawn in the same angle convention (``"cartesian"`` by default).
"""

from ._common import (CYCLIC_CMAP, add_colorbar, add_colorwheel, add_scalebar,
                      nice_length, show_image, symmetric_limits)
from .diagnostics import (plot_cages, plot_columns, plot_intensity_split,
                          plot_lattice_planes, plot_neighbor_cloud,
                          plot_neighbor_diagnostic, plot_pairs,
                          plot_reference_region)
from .displacement import (plot_displacement_map, plot_displacement_statistics,
                           plot_displacement_vectors, plot_magnitude_agreement)
from .strain import (plot_profiles, plot_strain_comparison, plot_strain_scatter,
                     plot_strain_tensor, plot_strain_triangulation)

__all__ = [
    "CYCLIC_CMAP", "add_colorbar", "add_colorwheel", "add_scalebar",
    "nice_length", "show_image", "symmetric_limits",
    "plot_columns", "plot_cages", "plot_pairs", "plot_intensity_split",
    "plot_neighbor_diagnostic", "plot_reference_region", "plot_lattice_planes",
    "plot_neighbor_cloud",
    "plot_displacement_vectors", "plot_displacement_map",
    "plot_displacement_statistics", "plot_magnitude_agreement",
    "plot_strain_scatter", "plot_strain_triangulation", "plot_strain_tensor",
    "plot_strain_comparison", "plot_profiles",
]
