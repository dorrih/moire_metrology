"""Tests for mesh generation."""

import numpy as np
import pytest

from moire_metrology.lattice import HexagonalLattice, MoireGeometry
from moire_metrology.mesh import MoireMesh


@pytest.fixture
def geometry():
    lat = HexagonalLattice(alpha=0.247)
    return MoireGeometry(lat, theta_twist=2.0)


class TestMeshGeneration:
    def test_basic_generation(self, geometry):
        mesh = MoireMesh.generate(geometry, pixel_size=0.5)
        assert mesh.n_vertices > 0
        assert mesh.n_triangles > 0
        assert mesh.points.shape[0] == 2

    def test_triangle_connectivity(self, geometry):
        mesh = MoireMesh.generate(geometry, pixel_size=0.5)
        # All triangle indices should be valid vertex indices
        assert np.all(mesh.triangles >= 0)
        assert np.all(mesh.triangles < mesh.n_vertices)

    def test_structured_mesh_counts(self, geometry):
        mesh = MoireMesh.generate(geometry, pixel_size=0.5)
        # Structured mesh: Nv = ns * nt, Nt = 2 * ns * nt
        assert mesh.n_vertices == mesh.ns * mesh.nt
        assert mesh.n_triangles == 2 * mesh.ns * mesh.nt

    def test_pixel_size_control(self, geometry):
        mesh_coarse = MoireMesh.generate(geometry, pixel_size=1.0, min_points=4)
        mesh_fine = MoireMesh.generate(geometry, pixel_size=0.3, min_points=4)
        assert mesh_fine.n_vertices > mesh_coarse.n_vertices

    def test_parametric_coords(self, geometry):
        mesh = MoireMesh.generate(geometry, pixel_size=0.5)
        s, t = mesh.parametric_coords()
        # All parametric coords should be in [0, 1)
        assert np.all(s >= -1e-10)
        assert np.all(s < 1.0 + 1e-10)
        assert np.all(t >= -1e-10)
        assert np.all(t < 1.0 + 1e-10)

    def test_n_scale(self, geometry):
        mesh1 = MoireMesh.generate(geometry, pixel_size=0.5, n_scale=1, min_points=4)
        mesh2 = MoireMesh.generate(geometry, pixel_size=0.5, n_scale=2, min_points=4)
        # n_scale=2 should have ~4x more vertices
        ratio = mesh2.n_vertices / mesh1.n_vertices
        assert 3.0 < ratio < 5.0
