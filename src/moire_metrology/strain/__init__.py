"""Strain extraction from moire patterns.

Two workflows:

1. **Point extraction** — given moire vector observables (lambda1, lambda2, phi1, phi2):
    from moire_metrology.strain import get_strain, get_strain_minimize_compression
    result = get_strain_minimize_compression(alpha1, alpha2, lambda1, lambda2, phi1, phi2)

2. **Spatial strain mapping** — from traced moire fringes:
    from moire_metrology.strain import FringeSet
    fringes = FringeSet.from_csv("fringes.csv")
    I_field, J_field = fringes.fit_registry_fields(order=8)
    strain_map = compute_strain_field(dIdx, dIdy, dJdx, dJdy, theta, theta0, alpha, delta)
"""

from .extraction import (
    StrainResult,
    compute_displacement_field,
    compute_strain_field,
    get_strain,
    get_strain_axis,
    get_strain_minimize_compression,
)
from .fringe import FringeLine, FringeSet
from .polynomial import RegistryField

__all__ = [
    "StrainResult",
    "get_strain",
    "get_strain_minimize_compression",
    "get_strain_axis",
    "compute_displacement_field",
    "compute_strain_field",
    "FringeLine",
    "FringeSet",
    "RegistryField",
]
