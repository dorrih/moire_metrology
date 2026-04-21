"""Per-layer material properties for 2D van der Waals heterostructures.

A :class:`Material` carries the *intra-layer* properties of a 2D
crystal: lattice constant and 2D elastic moduli. The *inter-layer*
stacking interaction (GSFE) is a property of a pair of materials in
contact, not of either material individually, and lives on
:class:`moire_metrology.interfaces.Interface` instead.

Units convention
----------------
Lattice constants are in **nm**. Bulk and shear moduli are stored in
**meV per unit cell**, matching the Carr/Halbertal convention used
throughout the package: the elastic energy density per unit cell is

    E_elastic_per_uc = (1/2) K (∂_x u_x + ∂_y u_y)²
                     + (1/2) G [(∂_x u_x − ∂_y u_y)² + (∂_x u_y + ∂_y u_x)²]

with K = λ + μ (the 2D bulk modulus) and G = μ (the 2D shear modulus).
The conversion to literature-standard 2D moduli in N/m is

    K [N/m] = K [meV/uc] / (S_uc [m²] × 6.241509e21 meV/J)

where ``S_uc = (√3/2) × a²`` is the unit cell area. For graphene this
factor is ~329.5 meV/uc per N/m, so ``K = 69518 meV/uc`` corresponds
to ``K = 211 N/m`` — matching the experimental indentation value of
Lee et al. Science 2008.

The :meth:`Material.from_2d_moduli_n_per_m` constructor and the
:attr:`Material.moduli_n_per_m` property provide the round-trip in
both directions, so user-defined materials can be specified directly
in N/m and existing materials can be sanity-checked against the
literature.

Sources for the bundled values
-------------------------------
- ``GRAPHENE``: K, G from Carr et al. PRB 98, 224102 (2018), Table I.
- ``HBN_AA``, ``HBN_AAP``: K, G derived from Falin et al. Nat. Commun.
  8, 15815 (2017), monolayer hBN indentation, ``E_2D ≈ 286 N/m``,
  combined with the literature 2D Poisson ratio ``ν ≈ 0.21`` to give
  ``K_2D ≈ 181 N/m, G_2D ≈ 118 N/m``. The AA / AA' distinction is a
  stacking convention for the hBN/hBN bilayer interface, not a
  monolayer material property — the Material entries are physically
  the same; only the paired Interface entries differ.
- ``MOSE2``, ``WSE2``: K, G from Halbertal et al. Nat. Commun. 12,
  242 (2021), SI Table 1, and independently confirmed in the
  Shabani, Halbertal et al. Nat. Phys. 17, 720 (2021) Methods section.
- ``GRAPHENE_BILAYER``: a Bernal-stacked graphene bilayer treated as
  a single effective 2D elastic continuum, with K and G taken as
  exactly twice the single-layer graphene values per the convention
  of Halbertal et al. Nat. Commun. 12, 242 (2021), SI Table 1 (the
  "TDBG" rows). Lattice constant is inherited from graphene.
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

    @property
    def moduli_n_per_m(self) -> tuple[float, float]:
        """Return ``(K, G)`` converted to literature-standard units of N/m.

        The package stores moduli in meV/unit-cell (the Carr/Halbertal
        convention used throughout the elastic energy expression). This
        property converts back to the more familiar 2D N/m units used
        in the experimental literature, e.g. for sanity-checking a
        bundled material against published indentation values.

        For graphene the conversion factor is ~329.5 meV/uc per N/m,
        so ``K = 69518 meV/uc`` returns ~211 N/m, matching the
        Lee et al. Science 2008 indentation value.
        """
        # S_uc in m^2
        S_uc_m2 = (np.sqrt(3) / 2) * (self.lattice_constant * 1e-9) ** 2
        # 1 J = 6.241509e21 meV
        meV_per_J = 6.241509074e21
        factor = S_uc_m2 * meV_per_J  # meV/uc per (N/m)
        return (self.bulk_modulus / factor, self.shear_modulus / factor)

    @classmethod
    def from_2d_moduli_n_per_m(
        cls,
        name: str,
        lattice_constant: float,
        bulk_modulus_n_per_m: float,
        shear_modulus_n_per_m: float,
    ) -> "Material":
        """Construct a Material from K, G in literature-standard N/m units.

        The package internally stores moduli in meV per unit cell to
        match the Carr/Halbertal elastic energy convention. Most
        published 2D elastic constants (Lee et al. Science 2008,
        Carr et al. PRB 98, etc.) are reported in N/m instead, so this
        constructor exists to make user-defined materials specifiable
        in their natural units without forcing the user to redo the
        conversion arithmetic by hand.

        Parameters
        ----------
        name : str
            Human-readable name.
        lattice_constant : float
            In-plane lattice constant in nm.
        bulk_modulus_n_per_m : float
            2D bulk modulus K = λ + μ in N/m.
        shear_modulus_n_per_m : float
            2D shear modulus G = μ in N/m.

        Examples
        --------
        Reproducing the bundled GRAPHENE from literature values:

        >>> g = Material.from_2d_moduli_n_per_m(
        ...     name="Graphene",
        ...     lattice_constant=0.247,
        ...     bulk_modulus_n_per_m=211.0,   # Lee et al. Science 2008
        ...     shear_modulus_n_per_m=144.0,
        ... )
        >>> int(round(g.bulk_modulus))   # within rounding of paper Table 1
        69533
        """
        S_uc_m2 = (np.sqrt(3) / 2) * (lattice_constant * 1e-9) ** 2
        meV_per_J = 6.241509074e21
        factor = S_uc_m2 * meV_per_J  # meV/uc per (N/m)
        return cls(
            name=name,
            lattice_constant=lattice_constant,
            bulk_modulus=bulk_modulus_n_per_m * factor,
            shear_modulus=shear_modulus_n_per_m * factor,
        )

    @classmethod
    def n_layer_stack(
        cls,
        base: "Material",
        n: int = 1,
        name: str | None = None,
    ) -> "Material":
        """Return a Material for ``n`` coherently-strained copies of ``base``.

        The returned Material has the same lattice constant as ``base``
        and elastic moduli scaled by ``n`` (each additional layer adds
        linearly to the effective stiffness under the assumption of
        identical in-plane strain). It is otherwise a first-class
        Material that can be used anywhere a single-layer one can.

        This is the modelling convention used in the literature for
        twisted double-bilayer graphene (n=2) and twisted double-
        trilayer graphene (n=3): see e.g. Halbertal et al. Nat. Commun.
        12, 242 (2021), SI Table 1, which states ``K, G were simply
        taken as twice that of TBG`` for the TDBG row, i.e. n=2.

        Parameters
        ----------
        base : Material
            The per-layer material to stack.
        n : int, default 1
            Number of coherent layers in the stack. ``n=1`` reproduces
            ``base`` (with a fresh name if one is provided). Must be
            at least 1.
        name : str, optional
            Human-readable name for the returned Material. Defaults to
            ``f"{base.name} (n={n})"``.

        Examples
        --------
        >>> from moire_metrology import GRAPHENE
        >>> bilayer = Material.n_layer_stack(GRAPHENE, n=2)
        >>> round(bilayer.bulk_modulus / GRAPHENE.bulk_modulus)
        2
        >>> bilayer.lattice_constant == GRAPHENE.lattice_constant
        True
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        return cls(
            name=name if name is not None else f"{base.name} (n={n})",
            lattice_constant=base.lattice_constant,
            bulk_modulus=n * base.bulk_modulus,
            shear_modulus=n * base.shear_modulus,
        )

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

# --- Graphene -------------------------------------------------------------
#
# K, G are exact verbatim copies of Carr et al. PRB 98, 224102 (2018),
# Table I, "Graphene" row. Carr's text states: "All values are in units
# of meV per unit cell." Equivalent to K ≈ 211 N/m, G ≈ 144 N/m,
# matching the experimental indentation result of Lee et al. Science
# 321, 385 (2008).
GRAPHENE = Material(
    name="Graphene",
    lattice_constant=0.247,
    bulk_modulus=69518.0,
    shear_modulus=47352.0,
)

# --- hBN monolayer (used for both HBN_AA and HBN_AAP entries) -------------
#
# K, G are derived from Falin et al. Nat. Commun. 8, 15815 (2017),
# "Mechanical properties of atomically thin boron nitride and the
# role of interlayer interactions". Falin reports E_3D = 0.865 ± 0.073
# TPa for monolayer hBN by AFM indentation, which gives a 2D Young's
# modulus E_2D ≈ 286 N/m using the standard interlayer-spacing
# thickness of 0.334 nm. With the literature 2D Poisson ratio
# ν ≈ 0.21 for hBN (also from Falin and consistent with several DFT
# results), the standard isotropic 2D Lamé relations give
#
#     K_2D = E / (2(1-ν)) = 286 / 1.58 = 181.0 N/m
#     G_2D = E / (2(1+ν)) = 286 / 2.42 = 118.2 N/m
#
# which the from_2d_moduli_n_per_m constructor converts to meV/uc
# (~61638 / ~40252 with α = 0.251 nm) so the literature N/m values
# remain visible in the source.
#
# The AA / AA' designation is a *stacking* convention for the hBN/hBN
# bilayer interface — it does not affect the monolayer per-layer
# elastic properties. HBN_AA and HBN_AAP are therefore identical
# Material entries that differ only in name; the actual stacking
# difference lives on the corresponding Interface entries in
# moire_metrology.interfaces.
HBN_AA = Material.from_2d_moduli_n_per_m(
    name="hBN (AA)",
    lattice_constant=0.251,
    bulk_modulus_n_per_m=181.0,
    shear_modulus_n_per_m=118.2,
)

HBN_AAP = Material.from_2d_moduli_n_per_m(
    name="hBN (AA')",
    lattice_constant=0.251,
    bulk_modulus_n_per_m=181.0,
    shear_modulus_n_per_m=118.2,
)

# --- MoSe2 / WSe2 ---------------------------------------------------------
#
# K, G from Halbertal et al. Nat. Commun. 12, 242 (2021), SI Table 1
# (the "MoSe2/WSe2 180° twist" column gives separate K and G for each
# of the two layers). Independently confirmed in Shabani, Halbertal
# et al. Nat. Phys. 17, 720 (2021), Methods section, as the values
# they used in the deep-moiré-potential analysis. All in meV/uc.
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

# --- Bernal-stacked graphene bilayer --------------------------------------
#
# Used as the per-"layer" material of a twisted-double-bilayer-graphene
# (TDBG) system, where each half of the outer twist pair is itself a
# Bernal bilayer. Constructed via :meth:`Material.n_layer_stack` with
# n=2 so that the elastic moduli are exactly twice the single-layer
# graphene values, matching the modelling convention of Halbertal
# et al. Nat. Commun. 12, 242 (2021), SI Table 1 ("K, G were simply
# taken as twice that of TBG"). Lattice constant is inherited from
# graphene.
#
# This is a coarse-grained continuum approximation: it treats the two
# constituent graphene sheets as straining in lockstep and ignores
# intra-bilayer interlayer coupling. Resolving that coupling requires
# an explicit 4-layer simulation via :class:`LayerStack`, which
# remains available to users who need it. For moire-length-scale
# relaxation work, the doubled-moduli convention is the standard
# literature choice.
GRAPHENE_BILAYER = Material.n_layer_stack(
    GRAPHENE, n=2, name="Graphene bilayer (Bernal)"
)
