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
    Material,
)
from .mesh import MoireMesh
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
    "MoireMesh",
    "RelaxationSolver",
    "SolverConfig",
]
