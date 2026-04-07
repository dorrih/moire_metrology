"""Moire Metrology: atomic relaxation and strain extraction for twisted 2D heterostructures.

Basic usage:
    from moire_metrology import RelaxationSolver, GRAPHENE

    solver = RelaxationSolver()
    result = solver.solve(
        material1=GRAPHENE,
        material2=GRAPHENE,
        theta_twist=2.0,
    )
    result.plot_stacking()
"""

from .gsfe import GSFESurface
from .lattice import HexagonalLattice, MoireGeometry
from .materials import (
    GRAPHENE,
    GRAPHENE_ON_HBN,
    HBN_AA,
    HBN_AAP,
    MOSE2,
    WSE2,
    Material,
)
from .mesh import MoireMesh, generate_finite_mesh
from .pinning import PinningMap, InteractivePinner
from .solver import RelaxationSolver, SolverConfig

__version__ = "0.1.0"

__all__ = [
    "GSFESurface",
    "HexagonalLattice",
    "MoireGeometry",
    "Material",
    "GRAPHENE",
    "GRAPHENE_ON_HBN",
    "HBN_AA",
    "HBN_AAP",
    "MOSE2",
    "WSE2",
    "MoireMesh",
    "generate_finite_mesh",
    "PinningMap",
    "InteractivePinner",
    "RelaxationSolver",
    "SolverConfig",
]
