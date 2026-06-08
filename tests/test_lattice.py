"""Tests for lattice geometry and moire calculations."""

import numpy as np
import pytest

from moire_metrology.lattice import HexagonalLattice, MoireGeometry, rotation_matrix


class TestRotationMatrix:
    def test_identity(self):
        R = rotation_matrix(0.0)
        np.testing.assert_allclose(R, np.eye(2), atol=1e-15)

    def test_90_degrees(self):
        R = rotation_matrix(90.0)
        np.testing.assert_allclose(R, [[0, -1], [1, 0]], atol=1e-15)

    def test_inverse(self):
        R = rotation_matrix(37.0)
        R_inv = rotation_matrix(-37.0)
        np.testing.assert_allclose(R @ R_inv, np.eye(2), atol=1e-15)


class TestHexagonalLattice:
    def test_basis_vectors_angle(self):
        lat = HexagonalLattice(alpha=0.247, theta0=0.0)
        dot = np.dot(lat.b1, lat.b2)
        # 60 degrees -> cos(60) = 0.5
        np.testing.assert_allclose(dot, 0.5, atol=1e-15)

    def test_basis_vectors_length(self):
        lat = HexagonalLattice(alpha=0.247, theta0=0.0)
        np.testing.assert_allclose(np.linalg.norm(lat.b1), 1.0, atol=1e-15)
        np.testing.assert_allclose(np.linalg.norm(lat.b2), 1.0, atol=1e-15)

    def test_unit_cell_area(self):
        lat = HexagonalLattice(alpha=0.247)
        expected = np.sqrt(3) / 2 * 0.247**2
        np.testing.assert_allclose(lat.unit_cell_area, expected, atol=1e-15)

    def test_basis_matrix_columns(self):
        lat = HexagonalLattice(alpha=0.247, theta0=0.0)
        B = lat.basis_matrix
        np.testing.assert_allclose(B[:, 0], 0.247 * lat.b1, atol=1e-15)
        np.testing.assert_allclose(B[:, 1], 0.247 * lat.b2, atol=1e-15)

    def test_reciprocal_matrix_orthogonality(self):
        """M @ (alpha * [b1|b2]) should give 2*pi*I."""
        lat = HexagonalLattice(alpha=0.247, theta0=15.0)
        M = lat.reciprocal_matrix
        B = lat.basis_matrix  # alpha * [b1|b2]
        product = M @ B
        np.testing.assert_allclose(product, 2 * np.pi * np.eye(2), atol=1e-12)


class TestMoireGeometry:
    def test_wavelength_formula(self):
        """For matched lattices, lambda = a / (2*sin(theta/2))."""
        alpha = 0.247
        for theta in [0.5, 1.0, 2.0, 5.0]:
            lat = HexagonalLattice(alpha=alpha)
            geom = MoireGeometry(lat, theta_twist=theta, delta=0.0)
            expected = alpha / (2 * np.sin(np.radians(theta / 2)))
            np.testing.assert_allclose(geom.wavelength, expected, rtol=0.02)

    def test_moire_vectors_exist(self):
        lat = HexagonalLattice(alpha=0.247)
        geom = MoireGeometry(lat, theta_twist=2.0)
        V1, V2 = geom.moire_vectors
        assert V1.shape == (2,)
        assert V2.shape == (2,)

    def test_moire_vectors_length(self):
        """V1 and V2 should have length close to the moire wavelength."""
        lat = HexagonalLattice(alpha=0.247)
        geom = MoireGeometry(lat, theta_twist=2.0)
        V1, V2 = geom.moire_vectors
        wl = geom.wavelength
        np.testing.assert_allclose(np.linalg.norm(V1), wl, rtol=0.1)
        np.testing.assert_allclose(np.linalg.norm(V2), wl, rtol=0.1)

    def test_stacking_phases_at_origin(self):
        """At origin with zero displacement, stacking phase should be zero."""
        lat = HexagonalLattice(alpha=0.247)
        geom = MoireGeometry(lat, theta_twist=2.0)
        v, w = geom.stacking_phases(np.array([0.0]), np.array([0.0]))
        np.testing.assert_allclose(v, 0.0, atol=1e-12)
        np.testing.assert_allclose(w, 0.0, atol=1e-12)

    def test_lattice_mismatch_wavelength(self):
        """For zero twist with mismatch, lambda ~ a/delta."""
        alpha = 0.247
        delta = 0.018  # ~1.8% like graphene/hBN
        lat = HexagonalLattice(alpha=alpha)
        geom = MoireGeometry(lat, theta_twist=0.0, delta=delta)
        expected = alpha / delta
        np.testing.assert_allclose(geom.wavelength, expected, rtol=0.1)

    def test_anisotropic_epsilon_breaks_symmetry(self):
        """Anisotropic heterostrain (diag(eps_x, 0)) makes |V1| != |V2|.

        The pure-twist case is isotropic (|V1| == |V2|); applying
        uniaxial strain along x at small twist should produce a
        large anisotropy ratio.
        """
        alpha = 0.246
        lat = HexagonalLattice(alpha=alpha)
        # Pure twist baseline: V1 and V2 same magnitude.
        g_iso = MoireGeometry(lat, theta_twist=0.01)
        np.testing.assert_allclose(
            np.linalg.norm(g_iso.V1), np.linalg.norm(g_iso.V2), rtol=1e-10)
        # Anisotropic: ε along x only.
        eps = np.array([[0.001, 0.0], [0.0, 0.0]])
        g_ani = MoireGeometry(lat, theta_twist=0.01, epsilon=eps)
        v1_mag = np.linalg.norm(g_ani.V1)
        v2_mag = np.linalg.norm(g_ani.V2)
        assert abs(v1_mag - v2_mag) / max(v1_mag, v2_mag) > 0.5, \
            "Anisotropic epsilon should noticeably break V1/V2 symmetry"

    def test_epsilon_isotropic_matches_scalar_delta(self):
        """epsilon = delta*I should give identical geometry to scalar delta."""
        alpha = 0.246
        delta = 0.005
        lat = HexagonalLattice(alpha=alpha)
        g_scalar = MoireGeometry(lat, theta_twist=0.5, delta=delta)
        g_tensor = MoireGeometry(
            lat, theta_twist=0.5, epsilon=delta * np.eye(2))
        np.testing.assert_allclose(g_scalar.V1, g_tensor.V1, atol=1e-12)
        np.testing.assert_allclose(g_scalar.V2, g_tensor.V2, atol=1e-12)
        np.testing.assert_allclose(g_scalar.Mu2, g_tensor.Mu2, atol=1e-12)
        # delta property should round-trip the isotropic case.
        assert g_tensor.delta == delta

    def test_epsilon_rejects_wrong_shape(self):
        """A non-(2,2) epsilon must raise ValueError at construction."""
        lat = HexagonalLattice(alpha=0.246)
        with pytest.raises(ValueError):
            MoireGeometry(lat, theta_twist=0.01,
                          epsilon=np.array([0.001, 0.0]))
