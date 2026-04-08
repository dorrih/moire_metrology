"""Interactive and programmatic tools for defining pinning sites.

Provides three ways to define where known stacking configurations are pinned:

1. **Programmatic** — PinningMap with pin_stacking() calls
2. **Interactive** — InteractivePinner with matplotlib click interface
3. **From file** — PinningMap.from_csv()

Example:
    from moire_metrology.pinning import PinningMap
    from moire_metrology import GRAPHENE, RelaxationSolver, SolverConfig
    from moire_metrology.lattice import HexagonalLattice, MoireGeometry
    from moire_metrology.mesh import MoireMesh
    from moire_metrology.discretization import Discretization

    geometry = MoireGeometry(HexagonalLattice(0.247), theta_twist=0.5)
    mesh = MoireMesh.generate(geometry, pixel_size=0.5)

    pins = PinningMap(mesh, geometry)
    pins.pin_stacking(x=10.0, y=20.0, stacking="AB", radius=3.0)
    pins.pin_stacking(x=30.0, y=15.0, stacking="BA", radius=3.0)

    disc = Discretization(mesh, geometry)
    conv = disc.build_conversion_matrices()
    constraints = pins.build_constraints(conv)

    solver = RelaxationSolver(SolverConfig(display=True))
    result = solver.solve(GRAPHENE, GRAPHENE, 0.5, constraints=constraints)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .discretization import ConversionMatrices, PinnedConstraints
from .lattice import MoireGeometry
from .mesh import MoireMesh


# Known stacking phases (v, w) for hexagonal lattices
STACKING_PHASES = {
    "AA": (0.0, 0.0),
    "AB": (2 * np.pi / 3, 2 * np.pi / 3),
    "BA": (4 * np.pi / 3, 4 * np.pi / 3),
}


@dataclass
class PinSite:
    """A single pinning site with known stacking."""

    x: float
    y: float
    stacking: str
    radius: float


class PinningMap:
    """Collection of pinning sites for constrained relaxation.

    Parameters
    ----------
    mesh : MoireMesh
        The computational mesh.
    geometry : MoireGeometry
        Moire geometry for computing stacking phases.
    """

    def __init__(self, mesh: MoireMesh, geometry: MoireGeometry):
        self.mesh = mesh
        self.geometry = geometry
        self.pins: list[PinSite] = []

    def pin_stacking(
        self,
        x: float,
        y: float,
        stacking: str,
        radius: float = 1.0,
    ):
        """Pin vertices near (x, y) to a known stacking configuration.

        Parameters
        ----------
        x, y : float
            Center of the pinning site (nm).
        stacking : str
            Stacking type: 'AA', 'AB', or 'BA'.
        radius : float
            Radius of the pinning region (nm). All mesh vertices within
            this radius are pinned.
        """
        if stacking not in STACKING_PHASES:
            raise ValueError(f"Unknown stacking '{stacking}'. Use one of: {list(STACKING_PHASES)}")
        self.pins.append(PinSite(x=x, y=y, stacking=stacking, radius=radius))

    def pin_vertices(
        self,
        vertex_indices: np.ndarray,
        stacking: str,
    ):
        """Pin specific vertices to a known stacking configuration.

        Parameters
        ----------
        vertex_indices : ndarray of int
            Mesh vertex indices to pin.
        stacking : str
            Stacking type: 'AA', 'AB', or 'BA'.
        """
        if stacking not in STACKING_PHASES:
            raise ValueError(f"Unknown stacking '{stacking}'. Use one of: {list(STACKING_PHASES)}")
        # Store as pins with radius=0 and the centroid position
        x_pts = self.mesh.points[0, vertex_indices]
        y_pts = self.mesh.points[1, vertex_indices]
        # Add a special pin that directly references vertex indices
        self.pins.append(PinSite(
            x=float(np.mean(x_pts)), y=float(np.mean(y_pts)),
            stacking=stacking, radius=-1.0,  # sentinel for vertex-indexed
        ))
        # Store the actual indices
        if not hasattr(self, '_vertex_pins'):
            self._vertex_pins = []
        self._vertex_pins.append((vertex_indices, stacking))

    def get_pinned_vertex_indices(self) -> np.ndarray:
        """Get all pinned vertex indices (union of all pin sites)."""
        all_pinned = set()

        for pin in self.pins:
            if pin.radius < 0:
                continue  # handled by _vertex_pins
            dist = np.sqrt(
                (self.mesh.points[0] - pin.x) ** 2
                + (self.mesh.points[1] - pin.y) ** 2
            )
            pinned = np.where(dist <= pin.radius)[0]
            all_pinned.update(pinned)

        if hasattr(self, '_vertex_pins'):
            for indices, _ in self._vertex_pins:
                all_pinned.update(indices)

        return np.array(sorted(all_pinned), dtype=int)

    def _compute_displacement_for_stacking(
        self, vertex_idx: int, stacking: str
    ) -> tuple[float, float]:
        """Compute the displacement (ux, uy) that produces the target stacking at a vertex.

        Solves: Mu @ [ux, uy] = -(v_target - v0, w_target - w0)
        """
        v_target, w_target = STACKING_PHASES[stacking]
        x = self.mesh.points[0, vertex_idx]
        y = self.mesh.points[1, vertex_idx]
        v0, w0 = self.geometry.stacking_phases(np.array([x]), np.array([y]))
        v0, w0 = float(v0[0]), float(w0[0])

        Mu = self.geometry.Mu1
        rhs = np.array([-(v_target - v0), -(w_target - w0)])
        u = np.linalg.solve(Mu, rhs)
        return float(u[0]), float(u[1])

    def build_constraints(
        self,
        conv: ConversionMatrices,
        nlayer1: int = 1,
        nlayer2: int = 1,
    ) -> PinnedConstraints:
        """Build PinnedConstraints from the current pin sites.

        Pins are applied to the interface layers (innermost of each stack).
        Both interface layers are pinned to produce the target stacking.

        Parameters
        ----------
        conv : ConversionMatrices
            From disc.build_conversion_matrices().
        nlayer1, nlayer2 : int
            Number of layers in each stack (needed for DOF offset calculation).

        Returns
        -------
        PinnedConstraints
        """
        Nv = conv.n_vertices
        nlayers_total = nlayer1 + nlayer2
        n_full = conv.n_sol

        # For each pinned vertex, compute displacement and set DOF values
        pinned_dof_map = {}  # dof_index -> value

        for pin in self.pins:
            if pin.radius < 0:
                continue  # handled below

            dist = np.sqrt(
                (self.mesh.points[0] - pin.x) ** 2
                + (self.mesh.points[1] - pin.y) ** 2
            )
            vertices = np.where(dist <= pin.radius)[0]

            for vi in vertices:
                ux, uy = self._compute_displacement_for_stacking(vi, pin.stacking)
                self._pin_vertex_dofs(pinned_dof_map, vi, ux, uy,
                                      Nv, nlayer1, nlayer2, nlayers_total)

        if hasattr(self, '_vertex_pins'):
            for indices, stacking in self._vertex_pins:
                for vi in indices:
                    ux, uy = self._compute_displacement_for_stacking(vi, stacking)
                    self._pin_vertex_dofs(pinned_dof_map, vi, ux, uy,
                                          Nv, nlayer1, nlayer2, nlayers_total)

        if not pinned_dof_map:
            raise ValueError("No vertices were pinned. Check pin positions and radii.")

        pinned_indices = np.array(sorted(pinned_dof_map.keys()), dtype=int)
        pinned_values = np.array([pinned_dof_map[i] for i in pinned_indices])
        free_indices = np.setdiff1d(np.arange(n_full), pinned_indices)

        return PinnedConstraints(
            free_indices=free_indices,
            pinned_indices=pinned_indices,
            pinned_values=pinned_values,
            n_free=len(free_indices),
            n_full=n_full,
        )

    def _pin_vertex_dofs(self, dof_map, vi, ux, uy, Nv, nlayer1, nlayer2, nlayers_total):
        """Pin the interface layer DOFs for vertex vi to displacement (ux, uy).

        For a bilayer, the relative displacement is split: layer1 gets +u/2,
        layer2 gets -u/2 (symmetric partition).
        """
        inner1_idx = nlayer1 - 1
        inner2_idx = nlayer1

        # Layer 1 inner: ux1 = +ux/2, uy1 = +uy/2
        dof_map[inner1_idx * Nv + vi] = ux / 2.0
        dof_map[nlayers_total * Nv + inner1_idx * Nv + vi] = uy / 2.0

        # Layer 2 inner: ux2 = -ux/2, uy2 = -uy/2
        dof_map[inner2_idx * Nv + vi] = -ux / 2.0
        dof_map[nlayers_total * Nv + inner2_idx * Nv + vi] = -uy / 2.0

    @classmethod
    def from_csv(cls, path: str | Path, mesh: MoireMesh, geometry: MoireGeometry) -> PinningMap:
        """Load pin sites from a CSV file.

        Expected format (with header): x, y, stacking, radius
        """
        path = Path(path)
        pins = cls(mesh, geometry)

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("x"):
                    continue
                parts = line.split(",")
                pins.pin_stacking(
                    x=float(parts[0]),
                    y=float(parts[1]),
                    stacking=parts[2].strip(),
                    radius=float(parts[3]),
                )

        return pins

    def save_csv(self, path: str | Path):
        """Save pin sites to a CSV file."""
        path = Path(path)
        with open(path, "w") as f:
            f.write("# x, y, stacking, radius\n")
            for pin in self.pins:
                if pin.radius >= 0:
                    f.write(f"{pin.x}, {pin.y}, {pin.stacking}, {pin.radius}\n")


class InteractivePinner:
    """Interactive matplotlib tool for placing pinning sites on a moire mesh.

    Usage:
        pinner = InteractivePinner(mesh, geometry)
        pinner.show()  # opens interactive window
        # Left-click to place pins, right-click to remove nearest
        # Press 1/2/3 to select AA/AB/BA stacking type
        # Press +/- to adjust radius
        # Close window when done
        constraints = pinner.get_constraints(conv)

    Parameters
    ----------
    mesh : MoireMesh
    geometry : MoireGeometry
    background_image : str or ndarray or None
        Path to an image file (e.g., PFM/STM data) to display behind the mesh,
        or a 2D numpy array. If None, shows the unrelaxed GSFE map.
    image_extent : tuple or None
        (xmin, xmax, ymin, ymax) for the background image in nm.
    """

    def __init__(
        self,
        mesh: MoireMesh,
        geometry: MoireGeometry,
        background_image=None,
        image_extent: tuple | None = None,
    ):
        self.pinning_map = PinningMap(mesh, geometry)
        self.mesh = mesh
        self.geometry = geometry
        self._bg_image = background_image
        self._bg_extent = image_extent
        self._current_stacking = "AB"
        self._current_radius = 2.0
        self._fig = None
        self._ax = None

    def show(self):
        """Open the interactive pinning window."""
        import matplotlib.pyplot as plt
        from matplotlib.tri import Triangulation

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        self._fig = fig
        self._ax = ax

        # Background
        if self._bg_image is not None:
            if isinstance(self._bg_image, (str, Path)):
                img = plt.imread(str(self._bg_image))
            else:
                img = self._bg_image
            extent = self._bg_extent or [
                self.mesh.points[0].min(), self.mesh.points[0].max(),
                self.mesh.points[1].min(), self.mesh.points[1].max(),
            ]
            ax.imshow(img, extent=extent, origin="lower", aspect="equal", alpha=0.7)
        else:
            # Show unrelaxed GSFE
            v0, w0 = self.geometry.stacking_phases(self.mesh.points[0], self.mesh.points[1])
            # Just use phase distance from AB as a proxy
            phase_dist = np.sqrt((v0 - 2*np.pi/3)**2 + (w0 - 2*np.pi/3)**2)
            tri = Triangulation(self.mesh.points[0], self.mesh.points[1], self.mesh.triangles)
            ax.tripcolor(tri, phase_dist, cmap="RdYlBu_r", alpha=0.5)

        ax.set_aspect("equal")
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        self._update_title()

        # Connect events
        fig.canvas.mpl_connect("button_press_event", self._on_click)
        fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._pin_artists = []
        plt.show()

    def _update_title(self):
        self._ax.set_title(
            f"Pinning: {self._current_stacking} | radius={self._current_radius:.1f} nm | "
            f"Pins: {len(self.pinning_map.pins)} | "
            f"Keys: 1=AA 2=AB 3=BA +/-=radius"
        )
        if self._fig is not None:
            self._fig.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes != self._ax:
            return

        if event.button == 1:  # Left click — place pin
            self.pinning_map.pin_stacking(
                x=event.xdata, y=event.ydata,
                stacking=self._current_stacking,
                radius=self._current_radius,
            )
            color = {"AA": "red", "AB": "blue", "BA": "green"}.get(self._current_stacking, "gray")
            circle = plt.Circle(
                (event.xdata, event.ydata), self._current_radius,
                fill=False, edgecolor=color, linewidth=2,
            )
            self._ax.add_patch(circle)
            self._ax.plot(event.xdata, event.ydata, "x", color=color, markersize=8)
            self._pin_artists.append(circle)

        elif event.button == 3:  # Right click — remove nearest pin
            if self.pinning_map.pins:
                dists = [
                    np.sqrt((p.x - event.xdata)**2 + (p.y - event.ydata)**2)
                    for p in self.pinning_map.pins
                ]
                idx = np.argmin(dists)
                self.pinning_map.pins.pop(idx)
                # Redraw (simplified — just clear and replot all)
                self._redraw_pins()

        self._update_title()

    def _on_key(self, event):
        if event.key == "1":
            self._current_stacking = "AA"
        elif event.key == "2":
            self._current_stacking = "AB"
        elif event.key == "3":
            self._current_stacking = "BA"
        elif event.key in ("+", "="):
            self._current_radius *= 1.2
        elif event.key in ("-", "_"):
            self._current_radius /= 1.2
        self._update_title()

    def _redraw_pins(self):
        """Redraw all pin markers."""
        import matplotlib.pyplot as plt
        # Remove old artists
        for a in self._pin_artists:
            a.remove()
        self._pin_artists.clear()

        for pin in self.pinning_map.pins:
            if pin.radius < 0:
                continue
            color = {"AA": "red", "AB": "blue", "BA": "green"}.get(pin.stacking, "gray")
            circle = plt.Circle((pin.x, pin.y), pin.radius, fill=False, edgecolor=color, linewidth=2)
            self._ax.add_patch(circle)
            self._ax.plot(pin.x, pin.y, "x", color=color, markersize=8)
            self._pin_artists.append(circle)
        self._fig.canvas.draw_idle()

    def get_constraints(
        self,
        conv: ConversionMatrices,
        nlayer1: int = 1,
        nlayer2: int = 1,
    ) -> PinnedConstraints:
        """Build constraints from the interactively placed pins."""
        return self.pinning_map.build_constraints(conv, nlayer1, nlayer2)
