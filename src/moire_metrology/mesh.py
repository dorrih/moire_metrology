"""Triangular mesh generation for moire unit cells.

Generates a regular grid on a parallelogram domain spanned by the moire
vectors V1, V2 (or multiples thereof), then triangulates with Delaunay.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lattice import MoireGeometry


@dataclass
class MoireMesh:
    """Triangular mesh on a moire unit cell parallelogram (or finite domain).

    Attributes
    ----------
    points : ndarray, shape (2, Nv)
        Vertex coordinates (x, y) in nm.
    triangles : ndarray, shape (Nt, 3)
        Triangle connectivity (vertex indices).
    V1, V2 : ndarray, shape (2,)
        Parallelogram edge vectors in nm. For a periodic mesh these are
        the moire lattice vectors. For a finite mesh they describe the
        bounding parallelogram of the domain (used by some plotting helpers).
    ns, nt : int
        Number of grid divisions along V1 and V2.
    is_periodic : bool
        True if the triangulation uses periodic wrap-around connectivity
        (the default, produced by ``MoireMesh.generate``). False if the
        mesh covers a finite domain with open boundaries (produced by
        ``generate_finite_mesh``). The discretization uses this flag to
        decide whether to apply lattice-vector wrap corrections to
        relative vertex positions when assembling differentiation
        matrices: periodic meshes need them, finite meshes don't.
    """

    points: np.ndarray
    triangles: np.ndarray
    V1: np.ndarray
    V2: np.ndarray
    ns: int
    nt: int
    n_scale: int
    is_periodic: bool = True
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
    *,
    n_cells_x: int | None = None,
    n_cells_y: int | None = None,
) -> MoireMesh:
    """Generate a non-periodic mesh covering multiple moire unit cells.

    Creates a parallelogram domain spanning n_cells moire vectors in
    each direction (or n_cells_x along V1 and n_cells_y along V2 when
    a rectangular extent is needed), with no periodic wrapping.
    Suitable for constrained relaxation.

    Parameters
    ----------
    geometry : MoireGeometry
        Moire geometry.
    n_cells : int
        Number of moire unit cells along each direction. Used as the
        default for both axes when n_cells_x / n_cells_y aren't given.
    pixel_size : float
        Target element size in nm.
    n_cells_x, n_cells_y : int or None
        Optional overrides for the number of moire cells along V1 and
        V2 respectively. If either is None, falls back to ``n_cells``.
        Use these to build a rectangular finite domain that's wider in
        one direction than the other (e.g. for a twist-gradient demo
        where you want many wavelengths along the gradient axis but
        only a few perpendicular to it).

    Returns
    -------
    MoireMesh
        Non-periodic mesh (boundary triangles do NOT wrap).
    """
    nx = n_cells if n_cells_x is None else n_cells_x
    ny = n_cells if n_cells_y is None else n_cells_y

    V1 = geometry.V1
    V2 = geometry.V2

    # Generate points on a grid covering nx * ny parallelograms
    ns = max(4, int(np.ceil(nx * np.linalg.norm(V1) / pixel_size)))
    nt = max(4, int(np.ceil(ny * np.linalg.norm(V2) / pixel_size)))

    s_vals = np.linspace(0, nx, ns + 1)
    t_vals = np.linspace(0, ny, nt + 1)

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
        V1=nx * V1,
        V2=ny * V2,
        ns=ns + 1,
        nt=nt + 1,
        n_scale=max(nx, ny),
        is_periodic=False,
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


def _hex_corners(V1: np.ndarray, V2: np.ndarray) -> np.ndarray:
    """Wigner-Seitz cell corners of the moire lattice (V1, V2) around origin.

    For a hexagonal lattice with primitive vectors V1, V2 at 60° apart and
    |V1| = |V2|, the WS cell is a regular hexagon with 6 corners at:

        c1 =  (V1 + V2) / 3         (an AB-stacking site)
        c2 = (-V1 + 2*V2) / 3       (a BA-stacking site)
        c3 = (-2*V1 + V2) / 3       (AB)
        c4 = -c1                    (BA)
        c5 = -c2                    (AB)
        c6 = -c3                    (BA)

    Returns shape (6, 2) array of corner positions, ordered counterclockwise.
    """
    c1 = (V1 + V2) / 3.0
    c2 = (-V1 + 2.0 * V2) / 3.0
    c3 = (-2.0 * V1 + V2) / 3.0
    return np.array([c1, c2, c3, -c1, -c2, -c3])


def _inside_hex(points: np.ndarray, V1: np.ndarray, V2: np.ndarray,
                 eps: float = 1e-9) -> np.ndarray:
    """Boolean mask: which 2D ``points`` (shape (..., 2)) lie inside (or on
    the boundary of) the Wigner-Seitz hexagon of the moire lattice (V1, V2).

    Test: |p · n| ≤ |n|² / 2  for n in {V1, V2, V2 - V1}  (three independent
    constraints; the other three are their negatives).
    """
    neighbors = np.array([V1, V2, V2 - V1])
    out = np.ones(points.shape[:-1], dtype=bool)
    for n in neighbors:
        n_sq_half = 0.5 * float(n @ n)
        out &= np.abs(points @ n) <= n_sq_half + eps
    return out


def generate_hex_periodic_mesh(
    geometry: MoireGeometry,
    pixel_size: float = 2.0,
    eps_boundary_frac: float = 0.05,
) -> MoireMesh:
    """Generate a hexagonal Wigner-Seitz periodic mesh centered on the AA
    stacking site (origin in our convention).

    The mesh has exact 6-fold rotational symmetry around the center: vertex
    positions are placed on a triangular sub-lattice through origin with
    spacing vectors V1/N, V2/N (so the sub-lattice has the same orientation
    as the moire lattice), then filtered to those inside the WS hexagon.
    The 6 WS corners are added explicitly.

    Use this together with the boundary identification helper
    :func:`identify_hex_periodic_boundary` and :class:`PeriodicPairConstraint`
    (3 pair sets, one for each pair of opposite edges) plus a
    :class:`PinnedConstraints` pinning the 6 corners at U=0 — see the
    online docs/example for a complete recipe.

    Compared to :func:`generate_finite_mesh` (parallelogram supercell, C2
    symmetry only), the hexagonal cell respects C3 symmetry around the AA
    site and the C3 orbit of AB / BA corner pins. This eliminates the
    direction-selection artifact that parallelogram supercells exhibit at
    low twist angles in problems with rich GSFE structure.

    Parameters
    ----------
    geometry : MoireGeometry
        Moire geometry defining V1, V2.
    pixel_size : float
        Target vertex spacing in nm.  The actual spacing is set by
        rounding |V1| / pixel_size to the nearest integer N, then using
        spacing |V1| / N (so the triangular sub-lattice tiles the cell
        exactly).
    eps_boundary_frac : float
        Tolerance for the "inside hex" test, in units of pixel_size.
        Vertices within this tolerance of the boundary are kept.

    Returns
    -------
    MoireMesh
        Hexagonal-cell mesh with ``is_periodic=False`` (open boundary;
        periodicity is enforced via PeriodicPairConstraint after boundary
        identification).  V1, V2 stored as the moire vectors (NOT the
        rhombus-cell vectors), so downstream code computing stacking
        phases from positions uses the correct geometry.
    """
    from scipy.spatial import Delaunay

    V1 = np.asarray(geometry.V1)
    V2 = np.asarray(geometry.V2)
    L = float(np.linalg.norm(V1))
    N = max(2, int(round(L / pixel_size)))
    g1 = V1 / N
    g2 = V2 / N

    # Search box for triangular-sub-lattice points
    R = N + 2
    ii, jj = np.meshgrid(np.arange(-R, R + 1), np.arange(-R, R + 1),
                          indexing='ij')
    coords = (ii[..., None] * g1 + jj[..., None] * g2).reshape(-1, 2)
    eps = eps_boundary_frac * pixel_size
    mask = _inside_hex(coords, V1, V2, eps=eps)
    interior = coords[mask]

    # Add the 6 WS corners explicitly so they are mesh vertices.
    corners = _hex_corners(V1, V2)
    all_pts = np.vstack([interior, corners])

    # Deduplicate (corner positions may coincide with sub-lattice points)
    decimals = max(0, int(-np.log10(0.01 * pixel_size)))
    key = np.round(all_pts, decimals=decimals)
    _, unique_idx = np.unique(key, axis=0, return_index=True)
    all_pts = all_pts[np.sort(unique_idx)]

    # Delaunay triangulation
    tri = Delaunay(all_pts)
    triangles = tri.simplices.astype(np.int64)

    return MoireMesh(
        points=all_pts.T,
        triangles=triangles,
        V1=V1,
        V2=V2,
        ns=N, nt=N, n_scale=1,
        is_periodic=False,
    )


def identify_hex_periodic_boundary(
    mesh: MoireMesh,
    tol: float = 1e-6,
) -> dict:
    """Identify boundary topology of a hexagonal-cell mesh built by
    :func:`generate_hex_periodic_mesh`.

    Returns a dict with:

    - ``"corners"`` — shape (6,) int array, mesh vertex indices for the 6
      WS corners, ordered c1, c2, c3, c4, c5, c6 (counterclockwise).
    - ``"edges"`` — dict mapping k ∈ {1, ..., 6} to the array of vertex
      indices on edge k, sorted along the c_{k} → c_{k+1} tangent
      direction.  Each edge includes its two endpoint corners.
    - ``"pairs"`` — list of three dicts, one per opposite-edge pair:
      ``{"source": k_src, "dest": k_dst, "translation": v, "src_indices":
      array, "dst_indices": array}``, where ``dst_indices[i] ≈
      src_indices[i] + translation`` in the parent moire lattice.

    Edge labelling (counterclockwise, edge k from c_k to c_{k+1}):
        k=1: c1→c2, outward normal +V2,        pair with k=4 (translation -V2)
        k=2: c2→c3, outward normal V2-V1,      pair with k=5 (-(V2-V1))
        k=3: c3→c4, outward normal -V1,        pair with k=6 (+V1)
        k=4: c4→c5, outward normal -V2,
        k=5: c5→c6, outward normal V1-V2,
        k=6: c6→c1, outward normal +V1.

    The three pairs (1↔4, 2↔5, 3↔6) cover ALL boundary vertices including
    corners (each corner is on two edges, so appears in two pair lists).
    """
    from scipy.spatial import cKDTree

    V1, V2 = np.asarray(mesh.V1), np.asarray(mesh.V2)
    points = mesh.points.T   # (Nv, 2)
    corners = _hex_corners(V1, V2)

    # Find which mesh vertex each corner corresponds to.
    tree = cKDTree(points)
    d_corner, corner_idxs = tree.query(corners, k=1)
    if d_corner.max() > tol:
        raise ValueError(
            f"Hex corners not found on mesh (max snap distance "
            f"{d_corner.max():.3e}). Mesh must include the 6 WS corners."
        )

    # Edge normals (n_k, sign so that p·n = +|n|²/2 on the edge)
    edge_defs = [
        (1, V2,      +1),
        (2, V2 - V1, +1),
        (3, V1,      -1),
        (4, V2,      -1),
        (5, V2 - V1, -1),
        (6, V1,      +1),
    ]
    edges = {}
    for k, n_vec, sign in edge_defs:
        target = sign * 0.5 * float(n_vec @ n_vec)
        proj = points @ n_vec
        on_edge = np.abs(proj - target) < tol * max(1.0, np.linalg.norm(n_vec))
        idxs = np.where(on_edge)[0]
        # Sort along the tangent from c_k to c_{k+1}
        t = corners[k % 6] - corners[(k - 1) % 6]
        sort_key = points[idxs] @ t
        edges[k] = idxs[np.argsort(sort_key)]

    # Build the three pairings.  For each pair (src, dst, translation),
    # match each src vertex to the dst vertex closest to src + translation.
    pair_defs = [
        (1, 4, -V2),
        (2, 5, V1 - V2),
        (6, 3, -V1),
    ]
    pairs = []
    for src_k, dst_k, translation in pair_defs:
        src_idxs = edges[src_k]
        dst_idxs = edges[dst_k]
        expected = points[src_idxs] + translation
        tree_dst = cKDTree(points[dst_idxs])
        d, kk = tree_dst.query(expected, k=1)
        matched = dst_idxs[kk]
        if d.max() > tol * max(1.0, np.linalg.norm(translation)):
            raise ValueError(
                f"Edge pair {src_k}↔{dst_k} failed to match: max mismatch "
                f"{d.max():.3e} (translation magnitude "
                f"{np.linalg.norm(translation):.3e})"
            )
        pairs.append({
            "source": src_k,
            "dest": dst_k,
            "translation": translation,
            "src_indices": src_idxs,
            "dst_indices": matched,
        })

    return {
        "corners": corner_idxs,
        "edges": edges,
        "pairs": pairs,
    }


def _circle_points(
    cx: float, cy: float, r: float, n: int,
) -> list[tuple[float, float]]:
    """Return *n* evenly spaced points on a circle."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]


def _connect_loop(start: int, end: int) -> list[tuple[int, int]]:
    """Facet list connecting indices start..end in a closed loop."""
    return [(i, i + 1) for i in range(start, end)] + [(end, start)]


def generate_custom_mesh(
    geometry: MoireGeometry,
    *,
    outer_boundary: str | list[tuple[float, float]] = "disk",
    outer_radius: float = 70.0,
    center: tuple[float, float] = (0.0, 0.0),
    holes: list[dict] | None = None,
    max_area: float = 2.0,
    min_angle: float = 25.0,
    boundary_density: int | None = None,
) -> MoireMesh:
    """Generate a mesh on an arbitrary 2D domain using meshpy.

    Produces an unstructured triangulation with smooth boundaries,
    suitable for constrained relaxation on non-rectangular domains
    and domains with interior holes.

    Parameters
    ----------
    geometry : MoireGeometry
        Moire geometry (provides V1, V2 for metadata).
    outer_boundary : ``"disk"`` or list of (x, y) tuples
        Outer domain boundary. ``"disk"`` creates a circle of
        radius ``outer_radius`` centred at ``center``.
    outer_radius : float
        Radius in nm (used only when ``outer_boundary="disk"``).
    center : tuple of float
        Centre of the disk in nm (default origin).
    holes : list of dict, optional
        Each dict describes a hole:
        ``{"center": (cx, cy), "radius": r}`` for a circular hole,
        or ``{"points": [(x1,y1), ...]}`` for an arbitrary polygon.
    max_area : float
        Maximum triangle area in nm² (controls mesh density).
    min_angle : float
        Minimum triangle angle in degrees (controls quality).
    boundary_density : int or None
        Number of boundary points per circle. If None, chosen
        automatically from ``outer_radius / sqrt(max_area)``.

    Returns
    -------
    MoireMesh
        Non-periodic mesh with ``is_periodic=False``.

    Examples
    --------
    Disk with a central hole::

        mesh = generate_custom_mesh(
            geom,
            outer_boundary="disk",
            outer_radius=70.0,
            holes=[{"center": (0, 0), "radius": 10.0}],
            max_area=1.5,
        )

    Hourglass (two pads connected by a neck)::

        # Define as a polygon
        mesh = generate_custom_mesh(
            geom,
            outer_boundary=hourglass_points,
            max_area=1.0,
        )
    """
    try:
        import meshpy.triangle as tri_mod
    except ImportError as e:
        raise ImportError(
            "meshpy is required for generate_custom_mesh. "
            "Install with: pip install meshpy"
        ) from e

    # --- Build outer boundary ---
    if outer_boundary == "disk":
        if boundary_density is None:
            boundary_density = max(32, int(2 * np.pi * outer_radius
                                           / np.sqrt(max_area)))
        cx, cy = center
        outer_pts = _circle_points(cx, cy, outer_radius, boundary_density)
    elif isinstance(outer_boundary, list):
        outer_pts = list(outer_boundary)
        if boundary_density is not None:
            raise ValueError(
                "boundary_density is only used with outer_boundary='disk'"
            )
    else:
        raise ValueError(
            f"outer_boundary must be 'disk' or a list of points, "
            f"got {outer_boundary!r}"
        )

    n_outer = len(outer_pts)
    all_points = list(outer_pts)
    all_facets = _connect_loop(0, n_outer - 1)
    hole_markers: list[tuple[float, float]] = []

    # --- Build hole boundaries ---
    if holes is not None:
        for h in holes:
            start_idx = len(all_points)
            if "radius" in h:
                hcx, hcy = h["center"]
                hr = h["radius"]
                n_hole = max(16, int(2 * np.pi * hr / np.sqrt(max_area)))
                hole_pts = _circle_points(hcx, hcy, hr, n_hole)
                hole_markers.append((hcx, hcy))
            elif "points" in h:
                hole_pts = list(h["points"])
                hx = sum(p[0] for p in hole_pts) / len(hole_pts)
                hy = sum(p[1] for p in hole_pts) / len(hole_pts)
                hole_markers.append((hx, hy))
            else:
                raise ValueError(
                    "Each hole must have 'radius'+'center' or 'points'"
                )
            end_idx = start_idx + len(hole_pts) - 1
            all_points.extend(hole_pts)
            all_facets.extend(_connect_loop(start_idx, end_idx))

    # --- Triangulate ---
    info = tri_mod.MeshInfo()
    info.set_points(all_points)
    info.set_facets(all_facets)
    if hole_markers:
        info.set_holes(hole_markers)

    built = tri_mod.build(
        info,
        max_volume=max_area,
        min_angle=min_angle,
    )

    pts_raw = np.array(built.points)     # (Nv, 2)
    tris_raw = np.array(built.elements)  # (Nt, 3)

    points = pts_raw.T  # (2, Nv) — MoireMesh convention

    # Identify boundary vertices (on outer and hole boundaries)
    outer_set = set(range(n_outer))
    hole_sets: list[set[int]] = []
    idx = n_outer
    if holes is not None:
        for h in holes:
            if "radius" in h:
                n_h = max(16, int(2 * np.pi * h["radius"]
                                  / np.sqrt(max_area)))
            else:
                n_h = len(h["points"])
            hole_sets.append(set(range(idx, idx + n_h)))
            idx += n_h

    # Compute bounding-parallelogram metadata
    V1 = geometry.V1
    V2 = geometry.V2
    x_all = points[0]
    y_all = points[1]
    extent = max(x_all.max() - x_all.min(), y_all.max() - y_all.min())
    n_scale = max(1, int(np.round(extent / geometry.wavelength)))

    return MoireMesh(
        points=points,
        triangles=tris_raw.astype(np.int64),
        V1=n_scale * V1,
        V2=n_scale * V2,
        ns=0,
        nt=0,
        n_scale=n_scale,
        is_periodic=False,
        _boundary_info={
            "outer_vertices": np.array(sorted(outer_set), dtype=np.int64),
            "hole_vertices": [
                np.array(sorted(s), dtype=np.int64) for s in hole_sets
            ],
        },
    )
