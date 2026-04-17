"""Tests for generate_custom_mesh (meshpy-based domain meshing).

Requires meshpy; all tests are skipped if meshpy is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import meshpy.triangle  # noqa: F401
    HAS_MESHPY = True
except ImportError:
    HAS_MESHPY = False

pytestmark = pytest.mark.skipif(not HAS_MESHPY, reason="meshpy not installed")

from moire_metrology import (  # noqa: E402
    GRAPHENE_GRAPHENE,
    MeanDisplacementConstraint,
    RelaxationSolver,
    SolverConfig,
    generate_custom_mesh,
)
from moire_metrology.discretization import Discretization, PinnedConstraints  # noqa: E402
from moire_metrology.lattice import HexagonalLattice, MoireGeometry  # noqa: E402


def _geom(theta=1.5):
    mat = GRAPHENE_GRAPHENE.bottom
    lat = HexagonalLattice(alpha=mat.lattice_constant)
    return MoireGeometry(lat, theta_twist=theta, delta=0.0)


class TestDisk:
    def test_basic_shape(self):
        geom = _geom()
        mesh = generate_custom_mesh(geom, outer_radius=30.0, max_area=3.0)
        assert mesh.n_vertices > 100
        assert mesh.n_triangles > 50
        assert not mesh.is_periodic
        # All vertices within outer radius (+ tolerance for boundary)
        r = np.hypot(mesh.points[0], mesh.points[1])
        assert r.max() <= 30.5

    def test_boundary_info(self):
        geom = _geom()
        mesh = generate_custom_mesh(geom, outer_radius=30.0, max_area=3.0)
        assert mesh._boundary_info is not None
        assert "outer_vertices" in mesh._boundary_info
        assert len(mesh._boundary_info["outer_vertices"]) > 10

    def test_mesh_density_scales_with_max_area(self):
        geom = _geom()
        coarse = generate_custom_mesh(geom, outer_radius=30.0, max_area=5.0)
        fine = generate_custom_mesh(geom, outer_radius=30.0, max_area=1.0)
        assert fine.n_vertices > coarse.n_vertices * 2


class TestDiskWithHole:
    def test_no_triangles_inside_hole(self):
        geom = _geom()
        mesh = generate_custom_mesh(
            geom, outer_radius=30.0,
            holes=[{"center": (0, 0), "radius": 5.0}],
            max_area=2.0,
        )
        # Triangle centroids must all be outside the hole
        tri = mesh.triangles
        cx = mesh.points[0][tri].mean(axis=1)
        cy = mesh.points[1][tri].mean(axis=1)
        r_centroid = np.hypot(cx, cy)
        assert r_centroid.min() > 4.5, (
            f"Triangle centroid at r={r_centroid.min():.2f} inside hole r=5.0"
        )

    def test_hole_boundary_vertices(self):
        geom = _geom()
        mesh = generate_custom_mesh(
            geom, outer_radius=30.0,
            holes=[{"center": (0, 0), "radius": 5.0}],
            max_area=2.0,
        )
        assert len(mesh._boundary_info["hole_vertices"]) == 1
        hole_verts = mesh._boundary_info["hole_vertices"][0]
        assert len(hole_verts) > 10
        # Hole boundary vertices should be near r=5.0
        r_hole = np.hypot(
            mesh.points[0][hole_verts],
            mesh.points[1][hole_verts],
        )
        np.testing.assert_allclose(r_hole, 5.0, atol=0.1)

    def test_multiple_holes(self):
        geom = _geom()
        mesh = generate_custom_mesh(
            geom, outer_radius=40.0,
            holes=[
                {"center": (-10, 0), "radius": 3.0},
                {"center": (10, 0), "radius": 3.0},
            ],
            max_area=2.0,
        )
        assert len(mesh._boundary_info["hole_vertices"]) == 2
        # No triangles inside either hole
        tri = mesh.triangles
        cx = mesh.points[0][tri].mean(axis=1)
        cy = mesh.points[1][tri].mean(axis=1)
        for hc in [(-10, 0), (10, 0)]:
            r = np.hypot(cx - hc[0], cy - hc[1])
            assert r.min() > 2.5


class TestPolygonBoundary:
    def test_square_domain(self):
        geom = _geom()
        L = 30.0
        square = [(-L, -L), (L, -L), (L, L), (-L, L)]
        mesh = generate_custom_mesh(
            geom, outer_boundary=square, max_area=3.0,
        )
        assert mesh.n_vertices > 100
        # All vertices within the square
        assert mesh.points[0].max() <= L + 0.1
        assert mesh.points[0].min() >= -L - 0.1


class TestRelaxation:
    def test_disk_relaxation_converges(self):
        """A relaxation solve on a custom disk mesh must converge."""
        geom = _geom(theta=1.5)
        mesh = generate_custom_mesh(
            geom, outer_radius=20.0, max_area=2.0,
        )
        Nv = mesh.n_vertices
        n_sol = 4 * Nv
        pinned = set(range(Nv, 2 * Nv)) | set(range(3 * Nv, 4 * Nv))
        pi = np.array(sorted(pinned), dtype=int)
        fi = np.array(sorted(set(range(n_sol)) - pinned), dtype=int)
        pc = PinnedConstraints(fi, pi, np.zeros(len(pi)), len(fi), n_sol)

        disc = Discretization(mesh, geom)
        conv = disc.build_conversion_matrices(nlayer1=1, nlayer2=1)
        mdc = MeanDisplacementConstraint.from_layer(conv, layer_idx=0)

        cfg = SolverConfig(
            method="newton", display=False,
            elastic_strain="cauchy",
            max_iter=200, gtol=1e-4,
        )
        r = RelaxationSolver(cfg).solve(
            moire_interface=GRAPHENE_GRAPHENE,
            theta_twist=1.5, delta=0.0,
            mesh=mesh, constraints=pc, mean_constraints=[mdc],
        )
        assert np.isfinite(r.total_energy)
        assert r.total_energy < r.unrelaxed_energy

    def test_disk_with_hole_relaxation_converges(self):
        """Relaxation on a disk with a hole must also converge."""
        geom = _geom(theta=1.5)
        mesh = generate_custom_mesh(
            geom, outer_radius=20.0,
            holes=[{"center": (0, 0), "radius": 3.0}],
            max_area=2.0,
        )
        Nv = mesh.n_vertices
        n_sol = 4 * Nv
        pinned = set(range(Nv, 2 * Nv)) | set(range(3 * Nv, 4 * Nv))
        pi = np.array(sorted(pinned), dtype=int)
        fi = np.array(sorted(set(range(n_sol)) - pinned), dtype=int)
        pc = PinnedConstraints(fi, pi, np.zeros(len(pi)), len(fi), n_sol)

        disc = Discretization(mesh, geom)
        conv = disc.build_conversion_matrices(nlayer1=1, nlayer2=1)
        mdc = MeanDisplacementConstraint.from_layer(conv, layer_idx=0)

        cfg = SolverConfig(
            method="newton", display=False,
            elastic_strain="cauchy",
            max_iter=200, gtol=1e-4,
        )
        r = RelaxationSolver(cfg).solve(
            moire_interface=GRAPHENE_GRAPHENE,
            theta_twist=1.5, delta=0.0,
            mesh=mesh, constraints=pc, mean_constraints=[mdc],
        )
        assert np.isfinite(r.total_energy)
        assert r.total_energy < r.unrelaxed_energy
