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

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on Python 3.10 only
    import tomli as tomllib


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

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Material":
        """Build a Material from a plain ``dict`` (e.g. parsed from TOML).

        Required keys: ``name``, ``lattice_constant``, ``bulk_modulus``,
        ``shear_modulus``. Extra keys raise a ``ValueError`` so that
        typos in user TOML files surface immediately instead of being
        silently ignored.
        """
        required = {"name", "lattice_constant", "bulk_modulus", "shear_modulus"}
        missing = required - set(data)
        if missing:
            raise ValueError(
                f"Material spec is missing required field(s): {sorted(missing)}. "
                f"A material needs name, lattice_constant (nm), bulk_modulus "
                f"(meV/uc), and shear_modulus (meV/uc)."
            )
        extra = set(data) - required
        if extra:
            raise ValueError(
                f"Material spec has unknown field(s): {sorted(extra)}. "
                f"Did you mean to put GSFE-related fields on an Interface "
                f"instead? Material only carries per-layer elastic properties."
            )
        return cls(
            name=str(data["name"]),
            lattice_constant=float(data["lattice_constant"]),
            bulk_modulus=float(data["bulk_modulus"]),
            shear_modulus=float(data["shear_modulus"]),
        )

    @classmethod
    def from_toml(cls, path: str | Path) -> "Material":
        """Load a Material from a TOML file.

        The file must contain a top-level ``[material]`` table with the
        four required fields::

            [material]
            name = "MyTMD"
            lattice_constant = 0.330      # nm
            bulk_modulus = 42000.0        # meV/uc
            shear_modulus = 28000.0       # meV/uc

        See :meth:`Material.from_dict` for the field semantics.
        """
        with open(path, "rb") as f:
            data = tomllib.load(f)
        if "material" not in data:
            raise ValueError(
                f"{path}: TOML file must contain a top-level [material] table."
            )
        return cls.from_dict(data["material"])


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
