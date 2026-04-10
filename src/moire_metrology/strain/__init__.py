"""Strain extraction from moire patterns.

Two workflows:

1. **Pointwise extraction** — given moire vector observables
   ``(lambda1, lambda2, phi1, phi2)`` for a single point::

        from moire_metrology.strain import get_strain_minimize_compression
        result = get_strain_minimize_compression(
            alpha1, alpha2, lambda1, lambda2, phi1_deg, phi2_deg,
        )

2. **Spatial strain mapping** — from traced moire fringe polylines::

        from moire_metrology.strain import (
            FringeSet, compute_strain_field, compute_displacement_field,
            convex_hull_mask,
        )
        fringes = FringeSet.from_matlab("points.mat")
        I_field, J_field = fringes.fit_registry_fields(order=11)
        strain = compute_strain_field(
            x, y, I_field, J_field, alpha1, alpha2, phi0_deg=-65.6,
        )
        ux, uy = compute_displacement_field(
            x, y, I_field, J_field, geometry, target_stacking="BA",
        )
"""

from .extraction import (
    StrainResult,
    compute_displacement_field,
    compute_strain_field,
    displacement_from_strain_field,
    get_strain,
    get_strain_axis,
    get_strain_minimize_compression,
    shear_strain_invariant,
)
from .fringe import FringeLine, FringeSet
from .polynomial import RegistryField
from .support import convex_hull_mask

__all__ = [
    "StrainResult",
    "get_strain",
    "get_strain_minimize_compression",
    "get_strain_axis",
    "shear_strain_invariant",
    "compute_displacement_field",
    "compute_strain_field",
    "displacement_from_strain_field",
    "FringeLine",
    "FringeSet",
    "RegistryField",
    "convex_hull_mask",
]
