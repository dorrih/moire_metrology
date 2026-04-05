"""Triangular mesh generation for moire unit cells.

Generates a regular grid on a parallelogram domain spanned by the moire
vectors V1, V2 (or multiples thereof), then triangulates with Delaunay.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay

from .lattice import MoireGeometry


@dataclass
class MoireMesh:
    """Triangular mesh on a moire unit cell parallelogram.

    Attributes
    ----------
    points : ndarray, shape (2, Nv)
        Vertex coordinates (x, y) in nm.
    triangles : ndarray, shape (Nt, 3)
        Triangle connectivity (vertex indices).
    V1, V2 : ndarray, shape (2,)
        Parallelogram edge vectors in nm.
    ns, nt : int
        Number of grid divisions along V1 and V2.
    boundary_edges : list of dict
        Each entry describes one parallelogram edge with keys:
        'vertices' (array of vertex indices), 'direction' ('s' or 't'),
        'value' (0.0 or 1.0).
    """

    points: np.ndarray
    triangles: np.ndarray
    V1: np.ndarray
    V2: np.ndarray
    ns: int
    nt: int
    n_scale: int
    _boundary_info: dict | None = None

    @property
    def n_vertices(self) -> int:
        return self.points.shape[1]

    @property
    def n_triangles(self) -> int:
        return self.triangles.shape[0]

    @classmethod
    def generate(
        cls,
        geometry: MoireGeometry,
        pixel_size: float = 0.2,
        n_scale: int = 1,
        min_points: int = 100,
        max_points: int = 500_000,
    ) -> MoireMesh:
        """Generate a mesh on the moire unit cell parallelogram.

        Parameters
        ----------
        geometry : MoireGeometry
            Defines V1, V2 moire vectors.
        pixel_size : float
            Target element size in nm.
        n_scale : int
            Number of moire unit cells along each direction.
        min_points, max_points : int
            Bounds on the number of grid points per direction.
        """
        V1 = n_scale * geometry.V1
        V2 = n_scale * geometry.V2

        # Determine grid resolution
        ns = max(min_points, int(np.ceil(np.linalg.norm(V1) / pixel_size)))
        nt = max(min_points, int(np.ceil(np.linalg.norm(V2) / pixel_size)))

        # Cap to avoid excessive memory
        total = ns * nt
        if total > max_points:
            scale_fac = np.sqrt(max_points / total)
            ns = int(ns * scale_fac)
            nt = int(nt * scale_fac)

        ns = max(ns, 4)
        nt = max(nt, 4)

        # Generate grid in parametric coordinates (s, t) in [0, 1)
        # Exclude s=1 and t=1 to avoid duplicating periodic boundary
        s_vals = np.arange(ns) / ns
        t_vals = np.arange(nt) / nt

        ss, tt = np.meshgrid(s_vals, t_vals, indexing="ij")
        ss_flat = ss.ravel()
        tt_flat = tt.ravel()

        # Map to real space: r = s * V1 + t * V2
        x = ss_flat * V1[0] + tt_flat * V2[0]
        y = ss_flat * V1[1] + tt_flat * V2[1]
        points = np.array([x, y])  # (2, Nv)

        # Build structured triangulation (two triangles per grid cell)
        triangles = _structured_triangulation(ns, nt)

        return cls(
            points=points,
            triangles=triangles,
            V1=V1,
            V2=V2,
            ns=ns,
            nt=nt,
            n_scale=n_scale,
        )

    def parametric_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the (s, t) parametric coordinates for each vertex.

        s, t in [0, 1), where r = s*V1 + t*V2.
        """
        # Solve [V1 | V2] @ [s; t] = [x; y]
        A = np.column_stack([self.V1, self.V2])
        st = np.linalg.solve(A, self.points)  # (2, Nv)
        return st[0], st[1]

    def get_boundary_vertices(self, tol: float = 1e-10) -> dict:
        """Identify vertices on each boundary of the parallelogram.

        Returns dict with keys: 's0', 's1', 't0', 't1' (edges at s=0, s~1, t=0, t~1),
        and 'corners' (vertices at corners).
        """
        if self._boundary_info is not None:
            return self._boundary_info

        s, t = self.parametric_coords()

        info = {
            "s0": np.where(np.abs(s) < tol)[0],
            "t0": np.where(np.abs(t) < tol)[0],
            # For periodic mesh, s=1 and t=1 are not in the grid,
            # so the "opposite" edges are s=0 and t=0 shifted by one period.
        }
        self._boundary_info = info
        return info


def _structured_triangulation(ns: int, nt: int) -> np.ndarray:
    """Build triangulation for a periodic ns x nt grid.

    Each grid cell (i, j) -> (i+1, j) -> (i, j+1) -> (i+1, j+1) is split
    into two triangles. Indices wrap around periodically.
    """
    triangles = []
    for i in range(ns):
        for j in range(nt):
            # Four corners of the grid cell (with periodic wrapping)
            i1 = (i + 1) % ns
            j1 = (j + 1) % nt
            v00 = i * nt + j
            v10 = i1 * nt + j
            v01 = i * nt + j1
            v11 = i1 * nt + j1
            # Two triangles
            triangles.append([v00, v10, v01])
            triangles.append([v10, v11, v01])

    return np.array(triangles, dtype=np.int64)


def generate_finite_mesh(
    geometry: MoireGeometry,
    n_cells: int = 3,
    pixel_size: float = 0.5,
) -> MoireMesh:
    """Generate a non-periodic mesh covering multiple moire unit cells.

    Creates a hexagonal domain of n_cells moire periods across, with
    no periodic wrapping. Suitable for constrained relaxation.

    Parameters
    ----------
    geometry : MoireGeometry
        Moire geometry.
    n_cells : int
        Number of moire unit cells across the domain diameter.
    pixel_size : float
        Target element size in nm.

    Returns
    -------
    MoireMesh
        Non-periodic mesh (boundary triangles do NOT wrap).
    """
    V1 = geometry.V1
    V2 = geometry.V2

    # Generate points on a grid covering n_cells x n_cells parallelograms
    ns = max(4, int(np.ceil(n_cells * np.linalg.norm(V1) / pixel_size)))
    nt = max(4, int(np.ceil(n_cells * np.linalg.norm(V2) / pixel_size)))

    s_vals = np.linspace(0, n_cells, ns + 1)
    t_vals = np.linspace(0, n_cells, nt + 1)

    ss, tt = np.meshgrid(s_vals, t_vals, indexing="ij")
    ss_flat = ss.ravel()
    tt_flat = tt.ravel()

    x = ss_flat * V1[0] + tt_flat * V2[0]
    y = ss_flat * V1[1] + tt_flat * V2[1]
    points = np.array([x, y])

    # Non-periodic triangulation (no wrapping)
    triangles = _structured_triangulation_open(ns + 1, nt + 1)

    return MoireMesh(
        points=points,
        triangles=triangles,
        V1=n_cells * V1,
        V2=n_cells * V2,
        ns=ns + 1,
        nt=nt + 1,
        n_scale=n_cells,
    )


def _structured_triangulation_open(ns: int, nt: int) -> np.ndarray:
    """Build triangulation for a NON-periodic ns x nt grid.

    Unlike the periodic version, indices do NOT wrap.
    """
    triangles = []
    for i in range(ns - 1):
        for j in range(nt - 1):
            v00 = i * nt + j
            v10 = (i + 1) * nt + j
            v01 = i * nt + (j + 1)
            v11 = (i + 1) * nt + (j + 1)
            triangles.append([v00, v10, v01])
            triangles.append([v10, v11, v01])

    return np.array(triangles, dtype=np.int64)
