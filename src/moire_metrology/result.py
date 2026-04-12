"""Result container for moire relaxation calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .interfaces import Interface
from .lattice import MoireGeometry
from .materials import Material
from .mesh import MoireMesh


@dataclass
class RelaxationResult:
    """Container for relaxation results with post-processing and plotting.

    Attributes
    ----------
    mesh : MoireMesh
        The computational mesh.
    geometry : MoireGeometry
        Moire geometry used in the calculation.
    moire_interface : Interface
        The twisted A-B interface that drove the calculation. Carries
        both materials (``moire_interface.top``, ``moire_interface.bottom``)
        and the GSFE coefficients of the moire stacking.
    top_interface : Interface or None
        Homobilayer interface used inside the top flake when
        ``nlayer1 > 1``. ``None`` for monolayer top flakes.
    bottom_interface : Interface or None
        Homobilayer interface used inside the bottom flake when
        ``nlayer2 > 1``. ``None`` for monolayer bottom flakes.
    displacement_x1, displacement_y1 : ndarray, shape (nlayer1, Nv)
        Displacement fields for the top flake (stack 1).
    displacement_x2, displacement_y2 : ndarray, shape (nlayer2, Nv)
        Displacement fields for the bottom flake (stack 2).
    total_energy : float
        Total relaxed energy in meV/uc.
    unrelaxed_energy : float
        Total unrelaxed energy in meV/uc.
    gsfe_map : ndarray, shape (Nv,)
        GSFE energy density at interface vertices (meV/nm^2).
    elastic_map1 : ndarray, shape (nlayer1, Nv)
        Elastic energy density per layer in the top flake (meV/nm^2).
    elastic_map2 : ndarray, shape (nlayer2, Nv)
        Elastic energy density per layer in the bottom flake (meV/nm^2).
    solution_vector : ndarray
        Raw optimizer solution vector.
    optimizer_result : object
        scipy.optimize result object.
    """

    mesh: MoireMesh
    geometry: MoireGeometry
    moire_interface: Interface
    top_interface: Interface | None
    bottom_interface: Interface | None
    displacement_x1: np.ndarray
    displacement_y1: np.ndarray
    displacement_x2: np.ndarray
    displacement_y2: np.ndarray
    total_energy: float
    unrelaxed_energy: float
    gsfe_map: np.ndarray
    elastic_map1: np.ndarray
    elastic_map2: np.ndarray
    solution_vector: np.ndarray
    optimizer_result: object

    @property
    def material1(self) -> Material:
        """Top flake material (stack 1).

        Convenience accessor — equivalent to ``self.moire_interface.top``.
        Stack-1 / stack-2 numbering follows the internal solver
        convention (stack 1 = top flake, stack 2 = bottom flake).
        """
        return self.moire_interface.top

    @property
    def material2(self) -> Material:
        """Bottom flake material (stack 2).

        Convenience accessor — equivalent to ``self.moire_interface.bottom``.
        """
        return self.moire_interface.bottom

    @property
    def nlayer1(self) -> int:
        return self.displacement_x1.shape[0]

    @property
    def nlayer2(self) -> int:
        return self.displacement_x2.shape[0]

    @property
    def energy_reduction(self) -> float:
        """Fractional energy reduction from relaxation."""
        if self.unrelaxed_energy == 0:
            return 0.0
        return (self.unrelaxed_energy - self.total_energy) / self.unrelaxed_energy

    @property
    def converged(self) -> bool:
        """Whether the optimizer reported successful convergence."""
        if self.optimizer_result is None:
            return True
        return getattr(self.optimizer_result, "success", True)

    @property
    def convergence_message(self) -> str:
        """Human-readable description of the convergence outcome."""
        if self.optimizer_result is None:
            return ""
        return str(getattr(self.optimizer_result, "message", ""))

    def local_twist(self, stack: int = 1, layer: int = 0) -> np.ndarray:
        """Compute local twist angle (degrees) at each vertex.

        local_twist = theta_twist + 0.5*(duy/dx - dux/dy) * 180/pi
        """
        from .discretization import Discretization

        disc = Discretization(self.mesh, self.geometry)

        if stack == 1:
            ux = self.displacement_x1[layer]
            uy = self.displacement_y1[layer]
        else:
            ux = self.displacement_x2[layer]
            uy = self.displacement_y2[layer]

        # Per-triangle derivatives
        dux_dy = disc.diff_mat_y @ ux
        duy_dx = disc.diff_mat_x @ uy

        # Local rotation = 0.5*(duy/dx - dux/dy)
        omega_t = 0.5 * (duy_dx - dux_dy)

        # Convert to vertex values
        omega_v = disc.triangle_to_vertex @ omega_t

        return self.geometry.theta_twist + np.degrees(omega_v)

    def plot_stacking(self, ax=None, n_tile: int = 2, **kwargs):
        """Plot the GSFE (stacking energy) map at the interface."""
        from .plotting import plot_scalar_field

        return plot_scalar_field(
            self.mesh,
            self.gsfe_map,
            ax=ax,
            n_tile=n_tile,
            title="Stacking energy (meV/nm$^2$)",
            cmap="RdYlBu_r",
            **kwargs,
        )

    def plot_elastic_energy(self, stack: int = 1, layer: int = 0, ax=None, n_tile: int = 2, **kwargs):
        """Plot elastic energy density for a given layer."""
        from .plotting import plot_scalar_field

        if stack == 1:
            data = self.elastic_map1[layer]
        else:
            data = self.elastic_map2[layer]

        return plot_scalar_field(
            self.mesh,
            data,
            ax=ax,
            n_tile=n_tile,
            title=f"Elastic energy stack {stack}, layer {layer} (meV/nm$^2$)",
            cmap="hot",
            **kwargs,
        )

    def plot_local_twist(self, stack: int = 1, layer: int = 0, ax=None, n_tile: int = 2, **kwargs):
        """Plot local twist angle map."""
        from .plotting import plot_scalar_field

        data = self.local_twist(stack=stack, layer=layer)
        return plot_scalar_field(
            self.mesh,
            data,
            ax=ax,
            n_tile=n_tile,
            title="Local twist angle (deg)",
            cmap="coolwarm",
            **kwargs,
        )

    def save(self, path: str):
        """Save result to a .npz file."""
        np.savez_compressed(
            path,
            points=self.mesh.points,
            triangles=self.mesh.triangles,
            V1=self.mesh.V1,
            V2=self.mesh.V2,
            ns=self.mesh.ns,
            nt=self.mesh.nt,
            n_scale=self.mesh.n_scale,
            displacement_x1=self.displacement_x1,
            displacement_y1=self.displacement_y1,
            displacement_x2=self.displacement_x2,
            displacement_y2=self.displacement_y2,
            total_energy=self.total_energy,
            unrelaxed_energy=self.unrelaxed_energy,
            gsfe_map=self.gsfe_map,
            elastic_map1=self.elastic_map1,
            elastic_map2=self.elastic_map2,
            solution_vector=self.solution_vector,
            theta_twist=self.geometry.theta_twist,
            delta=self.geometry.delta,
            alpha=self.geometry.lattice.alpha,
            theta0=self.geometry.lattice.theta0,
            moire_interface_name=self.moire_interface.name,
            material1_name=self.material1.name,
            material2_name=self.material2.name,
        )
