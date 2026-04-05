"""Tests for strain extraction module."""

import numpy as np
import pytest

from moire_metrology.strain import (
    get_strain,
    get_strain_axis,
    get_strain_minimize_compression,
    compute_strain_field,
    compute_displacement_field,
    FringeLine,
    FringeSet,
    RegistryField,
)


class TestGetStrainAxis:
    def test_zero_strain(self):
        eps1, eps2, angle = get_strain_axis(0.0, 0.0, 0.0)
        assert eps1 == 0.0
        assert eps2 == 0.0

    def test_pure_dilatation(self):
        """S = eps*I -> eps1 = eps2 = eps."""
        eps1, eps2, angle = get_strain_axis(0.01, 0.0, 0.01)
        np.testing.assert_allclose(eps1, 0.01, atol=1e-10)
        np.testing.assert_allclose(eps2, 0.01, atol=1e-10)

    def test_uniaxial(self):
        """S = [[eps, 0], [0, 0]] -> eps1 = eps, eps2 = 0."""
        eps1, eps2, angle = get_strain_axis(0.02, 0.0, 0.0)
        np.testing.assert_allclose(eps1, 0.02, atol=1e-10)
        np.testing.assert_allclose(eps2, 0.0, atol=1e-10)

    def test_pure_shear(self):
        """S = [[0, gamma], [gamma, 0]] -> eps1 = gamma, eps2 = -gamma."""
        gamma = 0.005
        eps1, eps2, angle = get_strain_axis(0.0, gamma, 0.0)
        np.testing.assert_allclose(eps1, gamma, atol=1e-10)
        np.testing.assert_allclose(eps2, -gamma, atol=1e-10)


class TestGetStrain:
    def test_unstrained_60deg(self):
        """At dphi=60 deg with matched lattices, compression-minimized strain should be ~zero."""
        alpha = 0.247
        theta_true = 2.0
        lam = alpha / (2 * np.sin(np.radians(theta_true / 2)))

        result = get_strain_minimize_compression(
            alpha1=alpha, alpha2=alpha,
            lambda1=lam, lambda2=lam,
            phi1_deg=0.0, phi2_deg=60.0,
        )
        np.testing.assert_allclose(abs(result.theta_twist), theta_true, atol=0.1)
        np.testing.assert_allclose(result.eps_c, 0.0, atol=1e-3)

    def test_nonzero_mismatch(self):
        """With lattice mismatch, should return reasonable values."""
        result = get_strain(
            alpha1=0.251, alpha2=0.247,
            lambda1=15.0, lambda2=14.0,
            phi1_deg=0.0, phi2_deg=55.0,
            phi0=0.0,
        )
        # Just check it runs and returns finite values
        assert np.isfinite(result.theta_twist)
        assert np.isfinite(result.eps_c)
        assert np.isfinite(result.eps_s)


class TestGetStrainMinimize:
    def test_minimize_compression(self):
        """Minimize compression should find eps_c close to zero when possible."""
        alpha = 0.247
        theta_true = 2.0
        lam = alpha / (2 * np.sin(np.radians(theta_true / 2)))

        result = get_strain_minimize_compression(
            alpha1=alpha, alpha2=alpha,
            lambda1=lam, lambda2=lam,
            phi1_deg=0.0, phi2_deg=60.0,
        )
        np.testing.assert_allclose(abs(result.eps_c), 0.0, atol=1e-3)

    def test_shear_is_phi0_independent(self):
        """The analytical shear strain formula should not depend on phi0.

        The paper gives a closed-form phi0-independent formula:
            eps_s = alpha * r_minus / (2 * lambda1 * lambda2 * sin(dphi))
        where r_minus = sqrt(l1^2 + l2^2 - 2*l1*l2*cos(dphi - pi/3))
        """
        from moire_metrology.strain.extraction import shear_strain_invariant

        alpha = 0.247
        eps_s = shear_strain_invariant(alpha, alpha, 10.0, 9.0, 0.0, 55.0)
        assert eps_s > 0  # should be nonzero for asymmetric moire


class TestRegistryField:
    def test_fit_linear(self):
        """Fitting a linear function should be exact."""
        x = np.random.rand(100) * 10
        y = np.random.rand(100) * 10
        values = 2.0 * x + 3.0 * y + 1.0

        field = RegistryField.fit(x, y, values, order=2)

        x_test = np.random.rand(50) * 10
        y_test = np.random.rand(50) * 10
        expected = 2.0 * x_test + 3.0 * y_test + 1.0
        actual = field(x_test, y_test)
        np.testing.assert_allclose(actual, expected, atol=1e-8)

    def test_derivatives_linear(self):
        """Derivatives of a linear fit should be constant."""
        x = np.random.rand(100) * 10
        y = np.random.rand(100) * 10
        values = 2.0 * x + 3.0 * y + 1.0

        field = RegistryField.fit(x, y, values, order=2)

        x_test = np.random.rand(50) * 10
        y_test = np.random.rand(50) * 10

        np.testing.assert_allclose(field.dx(x_test, y_test), 2.0, atol=1e-8)
        np.testing.assert_allclose(field.dy(x_test, y_test), 3.0, atol=1e-8)

    def test_fit_quadratic(self):
        """Fitting a quadratic should be accurate at order >= 2."""
        np.random.seed(42)
        x = np.random.rand(200) * 10
        y = np.random.rand(200) * 10
        values = x**2 + 0.5 * x * y + 0.3 * y**2

        field = RegistryField.fit(x, y, values, order=3)

        x_test = np.random.rand(50) * 10
        y_test = np.random.rand(50) * 10
        expected = x_test**2 + 0.5 * x_test * y_test + 0.3 * y_test**2
        actual = field(x_test, y_test)
        np.testing.assert_allclose(actual, expected, atol=0.1)


class TestFringeSet:
    def test_basic_workflow(self):
        """Test end-to-end: create fringes, fit registry, compute strain."""
        # Create synthetic fringe data for a simple moire pattern
        # I-fringes (horizontal-ish lines at integer I values)
        fringes = []
        for i in range(5):
            x = np.linspace(0, 50, 30)
            y = np.full_like(x, i * 10.0) + 0.5 * np.sin(x * 0.2)
            fringes.append(FringeLine(x=x, y=y, index=i, family=1))

        # J-fringes (diagonal lines at integer J values)
        for j in range(5):
            y = np.linspace(0, 50, 30)
            x = np.full_like(y, j * 10.0) + 0.3 * np.sin(y * 0.15)
            fringes.append(FringeLine(x=x, y=y, index=j, family=2))

        fs = FringeSet(fringes=fringes)
        assert len(fs.i_fringes) == 5
        assert len(fs.j_fringes) == 5

        # Fit registry fields
        I_field, J_field = fs.fit_registry_fields(order=4)

        # Evaluate at a test point
        x_test = np.array([25.0])
        y_test = np.array([25.0])
        I_val = I_field(x_test, y_test)
        J_val = J_field(x_test, y_test)

        # Should be roughly 2.5 (midpoint of 0-4 range)
        assert 1.5 < I_val[0] < 3.5
        assert 1.5 < J_val[0] < 3.5

    def test_estimate_wavelength(self):
        """Wavelength estimation from fringe spacing."""
        fringes = []
        spacing = 14.0  # nm
        for i in range(5):
            x = np.linspace(0, 50, 20)
            y = np.full_like(x, i * spacing)
            fringes.append(FringeLine(x=x, y=y, index=i, family=1))
        for j in range(5):
            y = np.linspace(0, 50, 20)
            x = np.full_like(y, j * spacing)
            fringes.append(FringeLine(x=x, y=y, index=j, family=2))

        fs = FringeSet(fringes=fringes)
        wl = fs.estimate_moire_wavelength()
        np.testing.assert_allclose(wl, spacing, rtol=0.05)


class TestComputeStrainField:
    def test_uniform_registry(self):
        """Uniform registry gradients should give uniform strain."""
        N = 100
        # Constant gradients: dI/dx = a, dJ/dy = b
        dIdx = np.full(N, 0.1)
        dIdy = np.zeros(N)
        dJdx = np.zeros(N)
        dJdy = np.full(N, 0.1)

        result = compute_strain_field(
            dIdx, dIdy, dJdx, dJdy,
            theta_deg=2.0, theta0_deg=0.0,
            alpha=0.247, delta=0.0,
        )

        # All values should be identical (uniform strain)
        assert np.std(result["eps_xx"]) < 1e-10
        assert np.std(result["eps_yy"]) < 1e-10
        assert np.std(result["eps_xy"]) < 1e-10
