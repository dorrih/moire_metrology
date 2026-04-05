"""Material database for 2D van der Waals heterostructures.

GSFE coefficients use the Carr convention (AB-referenced for centrosymmetric materials).
Units: lattice constants in nm, elastic moduli and GSFE coefficients in meV/unit cell.

The GSFE Fourier expansion is:
    V(v,w) = c0 + c1*(cos(v)+cos(w)+cos(v+w))
           + c2*(cos(v+2w)+cos(v-w)+cos(2v+w))
           + c3*(cos(2v)+cos(2w)+cos(2v+2w))
           + c4*(sin(v)+sin(w)-sin(v+w))
           + c5*(sin(2v+2w)-sin(2v)-sin(2w))

References:
    Zhou et al., PRB 92, 155438 (2015) — original GSFE parameterizations
    Carr et al., PRB 98, 224102 (2018) — convention used here
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Material:
    """A 2D material with elastic and interlayer interaction parameters.

    Parameters
    ----------
    name : str
        Human-readable name.
    lattice_constant : float
        In-plane lattice constant in nm.
    bulk_modulus : float
        2D bulk modulus K = lambda + mu in meV/unit cell.
    shear_modulus : float
        2D shear modulus G = mu in meV/unit cell.
    gsfe_coeffs : tuple[float, ...]
        GSFE Fourier coefficients (c0, c1, c2, c3, c4, c5) in meV/unit cell.
        For centrosymmetric materials (graphene, hBN-AA), c4 = c5 = 0.
    stacking_func : callable or None
        Function (k: int) -> (I, J) giving the natural Bernal stacking offset
        for layer k within a multi-layer stack. None for single-interface use.
    """

    name: str
    lattice_constant: float
    bulk_modulus: float
    shear_modulus: float
    gsfe_coeffs: tuple[float, ...] = field(repr=False)
    stacking_func: Callable[[int], tuple[float, float]] | None = field(
        default=None, repr=False
    )

    @property
    def unit_cell_area(self) -> float:
        """Unit cell area in nm^2."""
        return np.sqrt(3) / 2 * self.lattice_constant**2


def _zhou_to_carr(c_zhou: tuple[float, ...], alpha: float, ab_ref: bool = True) -> tuple[float, ...]:
    """Convert Zhou et al. (mJ/m^2) coefficients to Carr convention (meV/uc).

    Parameters
    ----------
    c_zhou : tuple
        (c0, c1, c2, c3, c4, c5) in mJ/m^2 from Zhou et al.
    alpha : float
        Lattice constant in nm.
    ab_ref : bool
        If True, transform to AB-referenced convention (for centrosymmetric materials).
    """
    e_charge = 1.60217662e-19  # C
    Suc = np.sqrt(3) / 2 * alpha**2  # nm^2
    fac = Suc / (1e18 * e_charge)  # mJ/m^2 -> meV/uc

    c0z, c1z, c2z, c3z, c4z, c5z = c_zhou

    if ab_ref:
        c0 = fac * c0z
        c1 = fac * (-0.5 * (c1z + np.sqrt(3) * c4z))
        c2 = fac * c2z
        c3 = fac * (-0.5 * (c3z - np.sqrt(3) * c5z))
        c4 = fac * (0.5 * (np.sqrt(3) * c1z - c4z))
        c5 = fac * 0.5 * (np.sqrt(3) * c3z + c5z)
    else:
        c0 = fac * c0z
        c1 = fac * c1z
        c2 = fac * c2z
        c3 = fac * c3z
        c4 = fac * c4z
        c5 = fac * c5z

    return (c0, c1, c2, c3, c4, c5)


def _graphene_stacking(k: int) -> tuple[float, float]:
    """Bernal (AB) stacking for graphene: alternating (1/3, 1/3) and (0, 0)."""
    val = (1 / 3) * ((-1) ** k)
    return (val, val)


def _hbn_aap_stacking(k: int) -> tuple[float, float]:
    """AA' stacking for hBN."""
    return (0.0, 0.0)


# --- Pre-built materials ---

# Graphene-graphene (Zhou et al. DFT-D2)
_gr_zhou = (21.336, -6.127, -1.128, 0.143, np.sqrt(3) * (-6.127), -np.sqrt(3) * 0.143)
_gr_coeffs = _zhou_to_carr(_gr_zhou, 0.247, ab_ref=True)

GRAPHENE = Material(
    name="Graphene",
    lattice_constant=0.247,
    bulk_modulus=8595.0,
    shear_modulus=5765.0,
    gsfe_coeffs=_gr_coeffs,
    stacking_func=_graphene_stacking,
)

# hBN AA stacking
_hbn_aa_zhou = (28.454, -7.160, -0.496, -0.339, np.sqrt(3) * (-7.160), -np.sqrt(3) * (-0.339))
_hbn_aa_coeffs = _zhou_to_carr(_hbn_aa_zhou, 0.251, ab_ref=True)

HBN_AA = Material(
    name="hBN (AA)",
    lattice_constant=0.251,
    bulk_modulus=8595.0,
    shear_modulus=5765.0,
    gsfe_coeffs=_hbn_aa_coeffs,
    stacking_func=_graphene_stacking,
)

# hBN AA' stacking (breaks inversion symmetry)
_hbn_aap_zhou = (31.584, -9.935, -0.918, 0.325, -7.848, 0.67)
_hbn_aap_coeffs = _zhou_to_carr(_hbn_aap_zhou, 0.251, ab_ref=False)

HBN_AAP = Material(
    name="hBN (AA')",
    lattice_constant=0.251,
    bulk_modulus=8595.0,
    shear_modulus=5765.0,
    gsfe_coeffs=_hbn_aap_coeffs,
    stacking_func=_hbn_aap_stacking,
)

# Graphene on hBN (heterostructure interface)
_gr_hbn_zhou = (39.222, -11.96, -0.748, -0.366, 1.640, 0.201)
_gr_hbn_coeffs = _zhou_to_carr(_gr_hbn_zhou, 0.247, ab_ref=False)

GRAPHENE_ON_HBN = Material(
    name="Graphene/hBN",
    lattice_constant=0.247,
    bulk_modulus=8595.0,
    shear_modulus=5765.0,
    gsfe_coeffs=_gr_hbn_coeffs,
)
