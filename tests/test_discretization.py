"""Tests for FEM discretization."""

import numpy as np
import pytest

from moire_metrology.discretization import PeriodicDiscretization
from moire_metrology.lattice import HexagonalLattice, MoireGeometry
from moire_metrology.mesh import MoireMesh


@pytest.fixture
def setup():
    lat = HexagonalLattice(alpha=0.247)
    geom = MoireGeometry(lat, theta_twist=2.0)
    mesh = MoireMesh.generate(geom, pixel_size=0.5)
    disc = PeriodicDiscretization(mesh, geom)
    return mesh, geom, disc


class TestDiffMatrices:
    def test_constant_field(self, setup):
        """Derivative of a constant field should be zero."""
        mesh, geom, disc = setup
        f = np.ones(mesh.n_vertices) * 3.7
        dfx = disc.diff_mat_x @ f
        dfy = disc.diff_mat_y @ f
        np.testing.assert_allclose(dfx, 0.0, atol=1e-10)
        np.testing.assert_allclose(dfy, 0.0, atol=1e-10)

    def test_periodic_sinusoidal_x(self, setup):
        """Test d/dx of a periodic function f = sin(2*pi*s) where s is parametric coord."""
        mesh, geom, disc = setup
        V1 = mesh.V1
        V2 = mesh.V2
        # Use parametric coordinates: r = s*V1 + t*V2
        s, t = mesh.parametric_coords()
        # f = sin(2*pi*s) is periodic in s with period 1
        f = np.sin(2 * np.pi * s)
        # df/dx: chain rule. ds/dx comes from inverting r = s*V1 + t*V2
        A = np.column_stack([V1, V2])
        Ainv = np.linalg.inv(A)
        # ds/dx = Ainv[0,0], ds/dy = Ainv[0,1]
        # Get per-triangle s values for cos evaluation
        s_tri = np.mean(s[mesh.triangles], axis=1)
        expected_dfx_tri = 2 * np.pi * np.cos(2 * np.pi * s_tri) * Ainv[0, 0]

        dfx = disc.diff_mat_x @ f
        # Exclude boundary-wrapping triangles (where s wraps from ~1 to ~0)
        s_range = np.ptp(s[mesh.triangles], axis=1)
        interior = s_range < 0.5  # non-wrapping triangles
        np.testing.assert_allclose(dfx[interior], expected_dfx_tri[interior], atol=0.1)

    def test_periodic_sinusoidal_y(self, setup):
        """Test d/dy of a periodic function f = sin(2*pi*t)."""
        mesh, geom, disc = setup
        V1 = mesh.V1
        V2 = mesh.V2
        s, t = mesh.parametric_coords()
        f = np.sin(2 * np.pi * t)
        A = np.column_stack([V1, V2])
        Ainv = np.linalg.inv(A)
        t_tri = np.mean(t[mesh.triangles], axis=1)
        expected_dfy = 2 * np.pi * np.cos(2 * np.pi * t_tri) * Ainv[1, 1]

        dfy = disc.diff_mat_y @ f
        np.testing.assert_allclose(dfy, expected_dfy, atol=0.5)

    def test_interior_linear_x(self, setup):
        """For interior triangles (not spanning boundary), d/dx of x should be 1."""
        mesh, geom, disc = setup
        f = mesh.points[0]  # f = x
        dfx = disc.diff_mat_x @ f

        # Identify interior triangles: all vertices close together
        tri_pts = mesh.points[:, mesh.triangles]  # (2, Nt, 3)
        x_range = np.ptp(tri_pts[0], axis=1)
        y_range = np.ptp(tri_pts[1], axis=1)
        wl = geom.wavelength
        interior = (x_range < 0.5 * wl) & (y_range < 0.5 * wl)

        np.testing.assert_allclose(dfx[interior], 1.0, atol=1e-8)


class TestAreas:
    def test_triangle_areas_positive(self, setup):
        mesh, geom, disc = setup
        areas = disc.triangle_areas
        assert np.all(areas > 0)

    def test_total_area(self, setup):
        """Total area should equal |V1 x V2| (parallelogram area)."""
        mesh, geom, disc = setup
        V1, V2 = mesh.V1, mesh.V2
        expected = abs(V1[0] * V2[1] - V1[1] * V2[0])
        actual = disc.total_area
        np.testing.assert_allclose(actual, expected, rtol=1e-10)

    def test_vertex_areas_sum(self, setup):
        mesh, geom, disc = setup
        total_v = np.sum(disc.vertex_areas)
        total_t = np.sum(disc.triangle_areas)
        np.testing.assert_allclose(total_v, total_t, rtol=1e-10)


class TestConversionMatrices:
    def test_bilayer_shapes(self, setup):
        mesh, geom, disc = setup
        conv = disc.build_conversion_matrices(nlayer1=1, nlayer2=1)
        Nv = mesh.n_vertices
        assert conv.n_sol == 4 * Nv  # 2 layers * 2 components * Nv
        assert conv.conv_x1.shape == (Nv, 4 * Nv)
        assert conv.conv_y1.shape == (Nv, 4 * Nv)
        assert conv.conv_x2.shape == (Nv, 4 * Nv)
        assert conv.conv_y2.shape == (Nv, 4 * Nv)

    def test_extraction(self, setup):
        """Conversion matrices should correctly extract displacement components."""
        mesh, geom, disc = setup
        conv = disc.build_conversion_matrices(nlayer1=1, nlayer2=1)
        Nv = mesh.n_vertices

        # Create a known solution vector
        U = np.zeros(conv.n_sol)
        ux1_expected = np.random.randn(Nv)
        U[:Nv] = ux1_expected  # ux for layer 1

        ux1_extracted = conv.conv_x1 @ U
        np.testing.assert_allclose(ux1_extracted, ux1_expected, atol=1e-15)
