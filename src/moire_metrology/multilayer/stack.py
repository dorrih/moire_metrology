"""Multi-layer stack configuration for relaxation calculations.

Provides a convenient way to define heterostructure stacks with
multiple layers and compute relaxation across the full stack.

Example
-------
.. code-block:: python

    from moire_metrology.interfaces import GRAPHENE_GRAPHENE
    from moire_metrology.multilayer import LayerStack

    # 5 layers of graphene on top of 3 layers of graphene
    stack = LayerStack(
        moire_interface=GRAPHENE_GRAPHENE,
        top_interface=GRAPHENE_GRAPHENE,
        bottom_interface=GRAPHENE_GRAPHENE,
        n_top=5,
        n_bottom=3,
        theta_twist=0.5,
    )
    result = stack.solve()

For a single-interface bilayer (``n_top = n_bottom = 1``) only
``moire_interface`` is required and the homobilayer interfaces can be
omitted; for multi-layer flakes the matching homobilayer interface is
required and the constructor will raise if it is missing or
inconsistent with the moire interface materials.
"""

from __future__ import annotations

from dataclasses import dataclass


from ..interfaces import Interface
from ..solver import RelaxationSolver, SolverConfig, _validate_flake_interfaces


@dataclass
class LayerStack:
    """A heterostructure stack of two flakes.

    Parameters
    ----------
    moire_interface : Interface
        The twisted interface between the bottom layer of the top flake
        and the top layer of the bottom flake. Carries both materials
        (``moire_interface.top``, ``moire_interface.bottom``) and the
        GSFE coefficients of the moire stacking.
    top_interface : Interface or None
        Required iff ``n_top > 1``. The homobilayer interface inside
        the top flake. Must satisfy ``top_interface.bottom is
        top_interface.top is moire_interface.top``.
    bottom_interface : Interface or None
        Required iff ``n_bottom > 1``. The homobilayer interface inside
        the bottom flake. Must satisfy ``bottom_interface.bottom is
        bottom_interface.top is moire_interface.bottom``.
    n_top, n_bottom : int
        Number of layers in each flake.
    theta_twist : float
        Twist angle between the two flakes (degrees).
    delta : float or None
        Lattice mismatch. If None, computed from material lattice
        constants.
    theta0 : float
        Lattice orientation angle (degrees).
    """

    moire_interface: Interface
    top_interface: Interface | None = None
    bottom_interface: Interface | None = None
    n_top: int = 1
    n_bottom: int = 1
    theta_twist: float = 0.0
    delta: float | None = None
    theta0: float = 0.0

    def __post_init__(self) -> None:
        # Catch missing/mismatched homobilayer interfaces at
        # construction time, not at solve() time.
        _validate_flake_interfaces(
            self.moire_interface,
            self.top_interface,
            self.bottom_interface,
            self.n_top,
            self.n_bottom,
        )

    @property
    def top(self):
        """Top flake material (convenience accessor)."""
        return self.moire_interface.top

    @property
    def bottom(self):
        """Bottom flake material (convenience accessor)."""
        return self.moire_interface.bottom

    @property
    def total_layers(self) -> int:
        return self.n_top + self.n_bottom

    @property
    def computed_delta(self) -> float:
        if self.delta is not None:
            return self.delta
        return self.top.lattice_constant / self.bottom.lattice_constant - 1.0

    def solve(
        self,
        config: SolverConfig | None = None,
        *,
        fix_top: bool = False,
        fix_bottom: bool = False,
    ):
        """Run relaxation on this stack.

        Parameters
        ----------
        config : SolverConfig or None
            Solver configuration. If None, uses defaults.
        fix_top : bool
            Pin all DOFs of the topmost layer (the topmost layer of the
            top flake) to zero. Use this to simulate a rigid bulk above
            the heterostructure.
        fix_bottom : bool
            Pin all DOFs of the bottommost layer (the deepest layer of
            the bottom flake) to zero. Use this to simulate a rigid bulk
            below the heterostructure — typical for a twisted flake on
            a thick substrate (e.g. graphene on graphite).

        Returns
        -------
        RelaxationResult
        """
        solver = RelaxationSolver(config)
        return solver.solve(
            moire_interface=self.moire_interface,
            top_interface=self.top_interface,
            bottom_interface=self.bottom_interface,
            n_top=self.n_top,
            n_bottom=self.n_bottom,
            theta_twist=self.theta_twist,
            delta=self.delta,
            theta0=self.theta0,
            fix_top=fix_top,
            fix_bottom=fix_bottom,
        )

    def describe(self) -> str:
        """Human-readable description of the stack."""
        lines = [
            f"LayerStack: {self.n_top}x {self.top.name} / {self.n_bottom}x {self.bottom.name}",
            f"  Moire interface: {self.moire_interface.name}",
            f"  Twist angle: {self.theta_twist:.4f} deg",
            f"  Lattice mismatch: {self.computed_delta:.6f}",
            f"  Total layers: {self.total_layers}",
        ]
        if self.top_interface is not None and self.top_interface.stacking_func is not None:
            offsets = [self.top_interface.stacking_func(k) for k in range(1, self.n_top)]
            lines.append(f"  Top stacking: {offsets}")
        if self.bottom_interface is not None and self.bottom_interface.stacking_func is not None:
            offsets = [self.bottom_interface.stacking_func(k) for k in range(1, self.n_bottom)]
            lines.append(f"  Bottom stacking: {offsets}")
        return "\n".join(lines)
