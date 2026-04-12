"""Moire Metrology: atomic relaxation and strain extraction for twisted 2D heterostructures.

Basic usage:
    from moire_metrology import RelaxationSolver
    from moire_metrology.interfaces import GRAPHENE_GRAPHENE

    solver = RelaxationSolver()
    result = solver.solve(
        moire_interface=GRAPHENE_GRAPHENE,
        theta_twist=2.0,
    )
    result.plot_stacking()
"""

from .gsfe import GSFESurface
from .interfaces import (
    BUNDLED_INTERFACES,
    GRAPHENE_GRAPHENE,
    GRAPHENE_HBN_INTERFACE,
    HBN_AA_HOMOBILAYER,
    HBN_AAP_HOMOBILAYER,
    MOSE2_WSE2_H_INTERFACE,
    Interface,
)
from .lattice import HexagonalLattice, MoireGeometry
from .materials import (
    GRAPHENE,
    HBN_AA,
    HBN_AAP,
    MOSE2,
    WSE2,
    Material,
)
from .mesh import MoireMesh, generate_finite_mesh
from .pinning import PinningMap, InteractivePinner
from .solver import RelaxationSolver, SolverConfig

__version__ = "0.5.0"

__all__ = [
    "GSFESurface",
    "HexagonalLattice",
    "MoireGeometry",
    "Material",
    "Interface",
    "GRAPHENE",
    "HBN_AA",
    "HBN_AAP",
    "MOSE2",
    "WSE2",
    "GRAPHENE_GRAPHENE",
    "HBN_AA_HOMOBILAYER",
    "HBN_AAP_HOMOBILAYER",
    "GRAPHENE_HBN_INTERFACE",
    "MOSE2_WSE2_H_INTERFACE",
    "BUNDLED_INTERFACES",
    "MoireMesh",
    "generate_finite_mesh",
    "PinningMap",
    "InteractivePinner",
    "RelaxationSolver",
    "SolverConfig",
]
