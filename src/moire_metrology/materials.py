"""Per-layer material properties for 2D van der Waals heterostructures.

A :class:`Material` carries the *intra-layer* properties of a 2D
crystal: lattice constant and 2D elastic moduli. The *inter-layer*
stacking interaction (GSFE) is a property of a pair of materials in
contact, not of either material individually, and lives on
:class:`moire_metrology.interfaces.Interface` instead.

Units: lattice constants in nm; bulk and shear moduli in meV per
unit cell (the Carr/Zhou convention).

.. note::
    The bulk and shear moduli for graphene and hBN bundled below
    (K = 8595, G = 5765 meV/uc) are taken from the Zhou et al. DFT-D2
    derivation. The MATLAB code this package was ported from used a
    different parameterization (K = 69518, G = 47352) whose source is
    not yet pinned down — see ``project_KG_discrepancy`` in the
    maintainer notes. The TMD entries (MoSe2, WSe2) use values from
    the Shabani et al. Nature Physics paper and are not part of that
    discrepancy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Material:
    """Per-layer properties of a single 2D material.

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
    """

    name: str
    lattice_constant: float
    bulk_modulus: float
    shear_modulus: float

    @property
    def unit_cell_area(self) -> float:
        """Unit cell area in nm^2."""
        return np.sqrt(3) / 2 * self.lattice_constant**2


# --- Bundled materials ---
#
# Each entry below is intentionally GSFE-free — the registry-dependent
# stacking energy is on :class:`Interface` in :mod:`interfaces`, not
# here. Pair these materials with a bundled or user-defined Interface
# when calling the solver.

GRAPHENE = Material(
    name="Graphene",
    lattice_constant=0.247,
    bulk_modulus=8595.0,
    shear_modulus=5765.0,
)

HBN_AA = Material(
    name="hBN (AA)",
    lattice_constant=0.251,
    bulk_modulus=8595.0,
    shear_modulus=5765.0,
)

HBN_AAP = Material(
    name="hBN (AA')",
    lattice_constant=0.251,
    bulk_modulus=8595.0,
    shear_modulus=5765.0,
)

MOSE2 = Material(
    name="MoSe2",
    lattice_constant=0.3288,
    bulk_modulus=40521.0,
    shear_modulus=26464.0,
)

WSE2 = Material(
    name="WSe2",
    lattice_constant=0.3282,
    bulk_modulus=43113.0,
    shear_modulus=30770.0,
)
