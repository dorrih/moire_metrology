"""Interlayer interfaces for moire heterostructures.

An :class:`Interface` describes the registry-dependent stacking energy
between two adjacent layers — the GSFE — together with the convention
for how the upper layer is offset relative to the lower layer when
multi-layer flakes are stacked. GSFE is physically a property of the
*pair* of materials in contact, not of either material individually,
so this module separates that concern from the per-material elastic
moduli that live in :mod:`moire_metrology.materials`.

The GSFE Fourier expansion follows the Carr et al. convention:

    V(v, w) = c0 + c1*(cos(v) + cos(w) + cos(v + w))
            + c2*(cos(v + 2w) + cos(v - w) + cos(2v + w))
            + c3*(cos(2v) + cos(2w) + cos(2v + 2w))
            + c4*(sin(v) + sin(w) - sin(v + w))
            + c5*(sin(2v + 2w) - sin(2v) - sin(2w))

For centrosymmetric homobilayers (e.g. graphene/graphene, hBN-AA) the
sin coefficients vanish (c4 = c5 = 0). Heterointerfaces with broken
inversion symmetry (e.g. MoSe2/WSe2 H-stacking, hBN AA') carry
non-zero c4, c5.

References
----------
- Zhou et al., PRB 92, 155438 (2015) — original GSFE parameterizations
  for graphene and hBN homo/heterointerfaces.
- Carr et al., PRB 98, 224102 (2018) — Fourier basis convention used
  here, including the Zhou->Carr coefficient transformation.
- Shabani, Halbertal et al., Nature Physics 17, 720 (2021) — GSFE
  parameters for the H-stacked MoSe2/WSe2 heterointerface (already in
  Carr basis, no transformation needed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .materials import (
    GRAPHENE,
    HBN_AA,
    HBN_AAP,
    MOSE2,
    WSE2,
    Material,
)


@dataclass(frozen=True)
class Interface:
    """The registry-dependent stacking interaction between two adjacent layers.

    GSFE characterizes a *pair* of materials in contact, not a single
    material. An :class:`Interface` couples two :class:`Material`
    instances with the GSFE Fourier coefficients that describe their
    stacking energy landscape, plus (for homobilayer interfaces) the
    natural Bernal stacking convention used when multiple layers are
    stacked within a flake.

    Parameters
    ----------
    name : str
        Human-readable name (e.g. "Graphene/Graphene", "MoSe2/WSe2 (H)").
    bottom : Material
        The lower layer of the pair. Boundary conditions like
        ``fix_bottom`` clamp the bottommost layer of the bottom flake.
    top : Material
        The upper layer of the pair. May equal ``bottom`` (homobilayer
        interface) or differ (heterointerface).
    gsfe_coeffs : tuple[float, ...]
        GSFE Fourier coefficients ``(c0, c1, c2, c3, c4, c5)`` in
        meV per unit cell, in the Carr basis (see module docstring).
        For centrosymmetric homobilayers, c4 = c5 = 0.
    stacking_func : callable or None
        ``(k: int) -> (I, J)`` returning the natural Bernal stacking
        offsets for layer ``k`` within a flake of more than one layer
        of this same material. ``None`` for heterointerfaces (where
        no Bernal convention applies) and for monolayer flake usage.
    reference : str
        Free-form citation string for the GSFE numbers, so the
        provenance travels with the data.
    """

    name: str
    bottom: Material
    top: Material
    gsfe_coeffs: tuple[float, ...] = field(repr=False)
    stacking_func: Callable[[int], tuple[float, float]] | None = field(
        default=None, repr=False
    )
    reference: str = ""

    @property
    def is_homobilayer(self) -> bool:
        """True iff the bottom and top materials are the same object."""
        return self.bottom is self.top


# ---------------------------------------------------------------------------
# Helpers used to construct the bundled interface entries below
# ---------------------------------------------------------------------------


def _zhou_to_carr(
    c_zhou: tuple[float, ...], alpha: float, ab_ref: bool = True
) -> tuple[float, ...]:
    """Convert Zhou et al. (mJ/m^2) coefficients to Carr convention (meV/uc).

    Parameters
    ----------
    c_zhou : tuple
        ``(c0, c1, c2, c3, c4, c5)`` in mJ/m^2 from Zhou et al. PRB 92.
    alpha : float
        Lattice constant in nm.
    ab_ref : bool
        If True, transform to AB-referenced convention used for
        centrosymmetric (homobilayer) materials. If False, use the
        non-centrosymmetric convention used for heterointerfaces and
        for hBN AA' stacking.
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
    """AA' stacking for hBN — every layer aligned, zero offset."""
    return (0.0, 0.0)


# ---------------------------------------------------------------------------
# Bundled interfaces
# ---------------------------------------------------------------------------

# --- Graphene/Graphene homobilayer (Zhou et al. DFT-D2) -------------------
_gr_zhou = (21.336, -6.127, -1.128, 0.143, np.sqrt(3) * (-6.127), -np.sqrt(3) * 0.143)
_gr_coeffs = _zhou_to_carr(_gr_zhou, 0.247, ab_ref=True)

GRAPHENE_GRAPHENE = Interface(
    name="Graphene/Graphene",
    bottom=GRAPHENE,
    top=GRAPHENE,
    gsfe_coeffs=_gr_coeffs,
    stacking_func=_graphene_stacking,
    reference="Zhou et al., PRB 92, 155438 (2015), DFT-D2",
)

# --- hBN AA homobilayer (Zhou et al. DFT-D2, AB-referenced) ---------------
_hbn_aa_zhou = (
    28.454, -7.160, -0.496, -0.339,
    np.sqrt(3) * (-7.160), -np.sqrt(3) * (-0.339),
)
_hbn_aa_coeffs = _zhou_to_carr(_hbn_aa_zhou, 0.251, ab_ref=True)

HBN_AA_HOMOBILAYER = Interface(
    name="hBN (AA) homobilayer",
    bottom=HBN_AA,
    top=HBN_AA,
    gsfe_coeffs=_hbn_aa_coeffs,
    stacking_func=_graphene_stacking,
    reference="Zhou et al., PRB 92, 155438 (2015), DFT-D2",
)

# --- hBN AA' homobilayer (breaks inversion symmetry, no AB ref) -----------
_hbn_aap_zhou = (31.584, -9.935, -0.918, 0.325, -7.848, 0.67)
_hbn_aap_coeffs = _zhou_to_carr(_hbn_aap_zhou, 0.251, ab_ref=False)

HBN_AAP_HOMOBILAYER = Interface(
    name="hBN (AA') homobilayer",
    bottom=HBN_AAP,
    top=HBN_AAP,
    gsfe_coeffs=_hbn_aap_coeffs,
    stacking_func=_hbn_aap_stacking,
    reference="Zhou et al., PRB 92, 155438 (2015), DFT-D2",
)

# --- Graphene/hBN heterointerface -----------------------------------------
#
# Replaces the v0.1.0 ``GRAPHENE_ON_HBN`` Material, which was a workaround
# for the v0.1.0 single-GSFE-per-Material limitation. The hBN polytype
# the Zhou et al. coefficients were fitted against is not stated in the
# original docstring; defaulting to ``HBN_AA`` here. TODO: verify against
# Zhou et al. PRB 92 and update if it's actually AA'.
_gr_hbn_zhou = (39.222, -11.96, -0.748, -0.366, 1.640, 0.201)
_gr_hbn_coeffs = _zhou_to_carr(_gr_hbn_zhou, 0.247, ab_ref=False)

GRAPHENE_HBN_INTERFACE = Interface(
    name="Graphene/hBN",
    bottom=HBN_AA,
    top=GRAPHENE,
    gsfe_coeffs=_gr_hbn_coeffs,
    stacking_func=None,
    reference="Zhou et al., PRB 92, 155438 (2015), DFT-D2",
)

# --- MoSe2/WSe2 H-stacked heterointerface ---------------------------------
#
# GSFE coefficients from Shabani, Halbertal et al., Nature Physics 17,
# 720 (2021). The Methods section reports them in the same Carr-style
# basis the package uses, so no Zhou->Carr transformation is needed.
#
# In an H-stacked MoSe2/WSe2 bilayer the global energy minimum is the
# MX' stacking; XX' is most unfavourable; MM' is a local minimum. The
# c4, c5 coefficients (sin terms) are non-zero, reflecting the broken
# inversion symmetry of the heterointerface. The lattice mismatch
# (~0.18%) drives a non-trivial moire pattern even at zero twist.
_mose2_wse2_h_gsfe = (42.6, 16.0, -2.7, -1.1, 3.7, 0.6)

MOSE2_WSE2_H_INTERFACE = Interface(
    name="MoSe2/WSe2 (H-stacked)",
    bottom=WSE2,
    top=MOSE2,
    gsfe_coeffs=_mose2_wse2_h_gsfe,
    stacking_func=None,
    reference="Shabani, Halbertal et al., Nat. Phys. 17, 720 (2021), "
              "doi:10.1038/s41567-021-01174-7",
)


# ---------------------------------------------------------------------------
# Bundled interface registry — convenience tuple of every interface above
# ---------------------------------------------------------------------------

BUNDLED_INTERFACES: tuple[Interface, ...] = (
    GRAPHENE_GRAPHENE,
    HBN_AA_HOMOBILAYER,
    HBN_AAP_HOMOBILAYER,
    GRAPHENE_HBN_INTERFACE,
    MOSE2_WSE2_H_INTERFACE,
)
