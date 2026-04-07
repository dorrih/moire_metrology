"""Multi-layer stack configuration for relaxation calculations.

Provides a convenient way to define heterostructure stacks with
multiple layers and compute relaxation across the full stack.

Example:
    from moire_metrology.multilayer import LayerStack
    from moire_metrology import GRAPHENE, HBN_AA, RelaxationSolver

    # 5-layer graphene on 3-layer hBN
    stack = LayerStack(
        top=GRAPHENE, n_top=5,
        bottom=HBN_AA, n_bottom=3,
        theta_twist=0.5,
    )
    result = stack.solve()
"""

from __future__ import annotations

from dataclasses import dataclass


from ..materials import Material
from ..solver import RelaxationSolver, SolverConfig


@dataclass
class LayerStack:
    """A heterostructure stack of two materials.

    Parameters
    ----------
    top : Material
        Top material (twisted layer).
    n_top : int
        Number of layers in the top stack.
    bottom : Material
        Bottom material (substrate).
    n_bottom : int
        Number of layers in the bottom stack.
    theta_twist : float
        Twist angle between the two stacks (degrees).
    delta : float or None
        Lattice mismatch. If None, computed from material lattice constants.
    theta0 : float
        Lattice orientation angle (degrees).
    """

    top: Material
    n_top: int
    bottom: Material
    n_bottom: int
    theta_twist: float
    delta: float | None = None
    theta0: float = 0.0

    @property
    def total_layers(self) -> int:
        return self.n_top + self.n_bottom

    @property
    def computed_delta(self) -> float:
        if self.delta is not None:
            return self.delta
        return self.top.lattice_constant / self.bottom.lattice_constant - 1.0

    def solve(self, config: SolverConfig | None = None):
        """Run relaxation on this stack.

        Parameters
        ----------
        config : SolverConfig or None
            Solver configuration. If None, uses defaults.

        Returns
        -------
        RelaxationResult
        """
        solver = RelaxationSolver(config)
        return solver.solve(
            material1=self.top,
            material2=self.bottom,
            theta_twist=self.theta_twist,
            nlayer1=self.n_top,
            nlayer2=self.n_bottom,
            delta=self.delta,
            theta0=self.theta0,
        )

    def describe(self) -> str:
        """Human-readable description of the stack."""
        lines = [
            f"LayerStack: {self.n_top}x {self.top.name} / {self.n_bottom}x {self.bottom.name}",
            f"  Twist angle: {self.theta_twist:.4f} deg",
            f"  Lattice mismatch: {self.computed_delta:.6f}",
            f"  Total layers: {self.total_layers}",
        ]
        if self.n_top > 1 and self.top.stacking_func is not None:
            offsets = [self.top.stacking_func(k) for k in range(1, self.n_top)]
            lines.append(f"  Top stacking: {offsets}")
        if self.n_bottom > 1 and self.bottom.stacking_func is not None:
            offsets = [self.bottom.stacking_func(k) for k in range(1, self.n_bottom)]
            lines.append(f"  Bottom stacking: {offsets}")
        return "\n".join(lines)
