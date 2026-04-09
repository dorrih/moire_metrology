"""Tests for the spatial strain extraction API.

These exercise the rewritten ``compute_strain_field`` and
``compute_displacement_field`` functions, which take ``RegistryField``
polynomial fits and produce per-point strain / displacement maps via
the validated paper-Fig-1 formulae (eq. 9 + per-point ``get_strain``
and the ``Mu @ u = (v0 - v_target)`` solve, respectively).
"""

from __future__ import annotations

import numpy as np

from moire_metrology.discretization import Discretization
from moire_metrology.lattice import HexagonalLattice, MoireGeometry
from moire_metrology.mesh import generate_finite_mesh
from moire_metrology.pinning import STACKING_PHASES
from moire_metrology.strain import (
    RegistryField,
    compute_displacement_field,
    compute_strain_field,
    convex_hull_mask,
    displacement_from_strain_field,
)


def _registry_from_geometry(
    geometry: MoireGeometry,
    *,
    fov_half_nm: float = 50.0,
    n_grid: int = 21,
    order: int = 2,
) -> tuple[RegistryField, RegistryField, np.ndarray, np.ndarray]:
    """Build I, J registry polynomials from a known MoireGeometry.

    The unrelaxed stacking phase ``(v0, w0) = M_phase @ r`` is a *linear*
    function of position, so a degree-2 polynomial fit reproduces it
    exactly. Dividing by 2π gives the integer-spaced registry indices
    ``(I, J)`` that the rest of the spatial-strain machinery expects.
    """
    xs = np.linspace(-fov_half_nm, fov_half_nm, n_grid)
    ys = np.linspace(-fov_half_nm, fov_half_nm, n_grid)
    Xg, Yg = np.meshgrid(xs, ys)
    xf = Xg.ravel()
    yf = Yg.ravel()

    v0, w0 = geometry.stacking_phases(xf, yf)
    I_vals = v0 / (2.0 * np.pi)
    J_vals = w0 / (2.0 * np.pi)

    I_field = RegistryField.fit(xf, yf, I_vals, order=order)
    J_field = RegistryField.fit(xf, yf, J_vals, order=order)
    return I_field, J_field, Xg, Yg


class TestComputeStrainField:
    def test_homostructure_rigid_twist(self):
        """For a homostructure (delta=0) rigid twist, the inversion
        recovers |theta| exactly and zero strain at every point.

        Note: ``MoireGeometry`` uses ``R(-theta_twist)`` for the twist
        rotation, while ``get_strain`` (and the paper) uses ``R(+theta)``.
        The recovered ``theta`` is therefore the negative of the
        ``MoireGeometry`` input. ``|theta|`` is the unambiguous quantity.
        """
        theta_in = 1.6  # deg
        alpha = 0.246  # graphene-like; anything works for delta=0

        lattice = HexagonalLattice(alpha=alpha, theta0=0.0)
        geom = MoireGeometry(lattice, theta_twist=theta_in, delta=0.0)

        I_field, J_field, _, _ = _registry_from_geometry(geom, order=2)

        # Sample on a smaller interior grid (avoid the fit boundary).
        xs = np.linspace(-30, 30, 7)
        Xq, Yq = np.meshgrid(xs, xs)

        result = compute_strain_field(
            Xq, Yq, I_field, J_field,
            alpha1=alpha, alpha2=alpha, phi0_deg=0.0,
        )

        np.testing.assert_allclose(np.abs(result["theta"]), theta_in, atol=1e-8)
        np.testing.assert_allclose(result["eps_c"], 0.0, atol=1e-10)
        np.testing.assert_allclose(result["eps_s"], 0.0, atol=1e-10)
        # The strain tensor is also zero for a rigid twist.
        np.testing.assert_allclose(result["S11"], 0.0, atol=1e-10)
        np.testing.assert_allclose(result["S12"], 0.0, atol=1e-10)
        np.testing.assert_allclose(result["S22"], 0.0, atol=1e-10)

    def test_uniform_field_is_uniform(self):
        """A linear registry field has constant gradients, so the
        recovered θ, ε_c, ε_s should all be spatially uniform."""
        lattice = HexagonalLattice(alpha=0.246, theta0=0.0)
        geom = MoireGeometry(lattice, theta_twist=2.0, delta=0.0)
        I_field, J_field, _, _ = _registry_from_geometry(geom, order=2)

        xs = np.linspace(-25, 25, 11)
        Xq, Yq = np.meshgrid(xs, xs)
        result = compute_strain_field(
            Xq, Yq, I_field, J_field,
            alpha1=0.246, alpha2=0.246, phi0_deg=0.0,
        )
        for key in ("theta", "eps_c", "eps_s"):
            assert np.std(result[key]) < 1e-10, key

    def test_output_shapes_match_query(self):
        """Output arrays should have the same shape as the query grid."""
        lattice = HexagonalLattice(alpha=0.246, theta0=0.0)
        geom = MoireGeometry(lattice, theta_twist=2.0, delta=0.0)
        I_field, J_field, _, _ = _registry_from_geometry(geom, order=2)

        Xq, Yq = np.meshgrid(np.linspace(-10, 10, 5), np.linspace(-10, 10, 4))
        result = compute_strain_field(
            Xq, Yq, I_field, J_field,
            alpha1=0.246, alpha2=0.246, phi0_deg=0.0,
        )
        for key in ("theta", "eps_c", "eps_s",
                    "S11", "S12", "S22",
                    "eps1", "eps2", "strain_angle",
                    "lambda1", "lambda2", "phi1_deg", "phi2_deg"):
            assert result[key].shape == Xq.shape, key


class TestComputeDisplacementField:
    def test_displacement_realizes_target_phase(self):
        """The IC must produce the phase field it was solving for.

        ``compute_displacement_field`` solves ``Mu @ u = v0 - v_target``
        per point, where ``v_target = 2π·I + v_offset``. After applying
        the solved ``u``, ``geom.stacking_phases(x, y, ux, uy)`` must
        equal ``v_target`` exactly. This nails down the linear-algebra
        sign convention against ``MoireGeometry.stacking_phases`` —
        which is the actual contract the relaxation solver consumes.
        """
        theta = 1.5  # deg
        delta = 0.00183
        alpha_substrate = 0.3282

        lattice = HexagonalLattice(alpha=alpha_substrate, theta0=0.0)
        geom = MoireGeometry(lattice, theta_twist=theta, delta=delta)
        I_field, J_field, _, _ = _registry_from_geometry(geom, order=2)

        xs = np.linspace(-25, 25, 9)
        Xq, Yq = np.meshgrid(xs, xs)

        for stacking, (v_off, w_off) in STACKING_PHASES.items():
            ux, uy = compute_displacement_field(
                Xq, Yq, I_field, J_field, geom,
                target_stacking=stacking, remove_mean=False,
            )
            v, w = geom.stacking_phases(
                Xq.ravel(), Yq.ravel(), ux.ravel(), uy.ravel(),
            )
            # The IC was built to satisfy
            #     v(r) = 2π · I_field(r) + v_off
            v_target = 2.0 * np.pi * I_field(Xq.ravel(), Yq.ravel()) + v_off
            w_target = 2.0 * np.pi * J_field(Xq.ravel(), Yq.ravel()) + w_off
            np.testing.assert_allclose(v, v_target, atol=1e-9, err_msg=stacking)
            np.testing.assert_allclose(w, w_target, atol=1e-9, err_msg=stacking)

    def test_target_stacking_changes_displacement_by_constant(self):
        """Switching target_stacking should add a constant offset to u
        (since v_target only changes by a constant). With remove_mean=False,
        the diff between two stackings is a single (ux, uy) constant."""
        lattice = HexagonalLattice(alpha=0.3282, theta0=0.0)
        geom = MoireGeometry(lattice, theta_twist=1.5, delta=0.00183)
        I_field, J_field, _, _ = _registry_from_geometry(geom, order=2)

        xs = np.linspace(-20, 20, 7)
        Xq, Yq = np.meshgrid(xs, xs)

        ux_aa, uy_aa = compute_displacement_field(
            Xq, Yq, I_field, J_field, geom,
            target_stacking="AA", remove_mean=False,
        )
        ux_ba, uy_ba = compute_displacement_field(
            Xq, Yq, I_field, J_field, geom,
            target_stacking="BA", remove_mean=False,
        )
        dux = ux_ba - ux_aa
        duy = uy_ba - uy_aa
        assert np.std(dux) < 1e-12
        assert np.std(duy) < 1e-12

    def test_remove_mean_zeroes_translation_gauge(self):
        """remove_mean=True should produce zero-mean ux, uy."""
        lattice = HexagonalLattice(alpha=0.3282, theta0=0.0)
        geom = MoireGeometry(lattice, theta_twist=1.5, delta=0.00183)
        I_field, J_field, _, _ = _registry_from_geometry(geom, order=2)

        xs = np.linspace(-20, 20, 7)
        Xq, Yq = np.meshgrid(xs, xs)
        ux, uy = compute_displacement_field(
            Xq, Yq, I_field, J_field, geom, target_stacking="BA",
            remove_mean=True,
        )
        assert abs(ux.mean()) < 1e-12
        assert abs(uy.mean()) < 1e-12

    def test_unknown_stacking_raises(self):
        lattice = HexagonalLattice(alpha=0.3282, theta0=0.0)
        geom = MoireGeometry(lattice, theta_twist=1.5, delta=0.00183)
        I_field, J_field, _, _ = _registry_from_geometry(geom, order=2)

        import pytest
        with pytest.raises(ValueError, match="Unknown stacking"):
            compute_displacement_field(
                np.array([0.0]), np.array([0.0]),
                I_field, J_field, geom, target_stacking="BOGUS",
            )


class TestDisplacementFromStrainField:
    """Round-trip tests for the gradient-integration IC builder.

    The helper takes target ``(δθ(r), S(r))`` and integrates the
    implied ``∂u/∂r`` field on the FEM mesh. For a target ``u`` field
    that is *exactly affine* (constant gradients), the integrator
    should reproduce ``u`` exactly up to a global translation gauge
    constant — there's nothing to least-squares away.
    """

    def _build_mesh_disc(self, theta_avg_deg: float = 1.5):
        """A small finite mesh + discretization for these tests."""
        lattice = HexagonalLattice(alpha=0.3282, theta0=0.0)
        geom = MoireGeometry(lattice, theta_twist=theta_avg_deg, delta=0.00183)
        mesh = generate_finite_mesh(geom, n_cells=4, pixel_size=2.0)
        disc = Discretization(mesh, geom)
        return mesh, disc, geom

    def test_pure_strain_no_rotation(self):
        """Affine `u(r)` with pure strain (no local twist deviation):
        ux = a*x + b*y, uy = b*x + d*y. Symmetric ∂u/∂r → δθ = 0."""
        mesh, disc, _ = self._build_mesh_disc()
        a, b, d = 0.001, 0.0003, -0.0005

        x = mesh.points[0]
        y = mesh.points[1]
        ux_target = a * x + b * y
        uy_target = b * x + d * y

        Nv = mesh.n_vertices
        S11 = np.full(Nv, a)
        S22 = np.full(Nv, d)
        S12 = np.full(Nv, b)
        theta = np.full(Nv, 1.5)  # = theta_avg, so δθ = 0

        ux, uy = displacement_from_strain_field(
            disc,
            theta_deg=theta, theta_avg_deg=1.5,
            S11=S11, S12=S12, S22=S22,
            pin_vertex=0,
        )
        # Match up to a global translation: align by setting
        # u(pin_vertex) of the target to zero.
        ux_aligned = ux_target - ux_target[0]
        uy_aligned = uy_target - uy_target[0]
        np.testing.assert_allclose(ux, ux_aligned, atol=1e-9)
        np.testing.assert_allclose(uy, uy_aligned, atol=1e-9)

    def test_pure_rotation_no_strain(self):
        """Affine u with pure rotation: ux = -ω*y, uy = +ω*x.
        Antisymmetric ∂u/∂r → S = 0, δθ = ω (in radians)."""
        mesh, disc, _ = self._build_mesh_disc()
        omega_rad = np.radians(0.05)  # 0.05° local twist deviation

        x = mesh.points[0]
        y = mesh.points[1]
        ux_target = -omega_rad * y
        uy_target = +omega_rad * x

        Nv = mesh.n_vertices
        S11 = np.zeros(Nv)
        S22 = np.zeros(Nv)
        S12 = np.zeros(Nv)
        theta_avg_deg = 1.5
        theta = np.full(Nv, theta_avg_deg + 0.05)  # δθ = +0.05°

        ux, uy = displacement_from_strain_field(
            disc,
            theta_deg=theta, theta_avg_deg=theta_avg_deg,
            S11=S11, S12=S12, S22=S22,
            pin_vertex=0,
        )
        ux_aligned = ux_target - ux_target[0]
        uy_aligned = uy_target - uy_target[0]
        np.testing.assert_allclose(ux, ux_aligned, atol=1e-9)
        np.testing.assert_allclose(uy, uy_aligned, atol=1e-9)

    def test_general_affine(self):
        """Affine u with both strain and rotation. Verifies that the
        integrator correctly disentangles symmetric and antisymmetric
        gradient components and pins down the eps=0.5 partition factor."""
        mesh, disc, _ = self._build_mesh_disc()
        # u_x = a*x + b*y, u_y = c*x + d*y
        # symmetric: S11=a, S22=d, S12=(b+c)/2
        # rotation: δθ = (c - b)/2
        a, b, c, d = 0.0008, 0.0003, -0.0002, -0.0006

        x = mesh.points[0]
        y = mesh.points[1]
        ux_target = a * x + b * y
        uy_target = c * x + d * y

        Nv = mesh.n_vertices
        S11 = np.full(Nv, a)
        S22 = np.full(Nv, d)
        S12 = np.full(Nv, 0.5 * (b + c))
        delta_theta_rad = 0.5 * (c - b)
        delta_theta_deg = np.degrees(delta_theta_rad)
        theta_avg_deg = 1.5
        theta = np.full(Nv, theta_avg_deg + delta_theta_deg)

        ux, uy = displacement_from_strain_field(
            disc,
            theta_deg=theta, theta_avg_deg=theta_avg_deg,
            S11=S11, S12=S12, S22=S22,
            pin_vertex=0,
        )
        ux_aligned = ux_target - ux_target[0]
        uy_aligned = uy_target - uy_target[0]
        np.testing.assert_allclose(ux, ux_aligned, atol=1e-9)
        np.testing.assert_allclose(uy, uy_aligned, atol=1e-9)


class TestConvexHullMask:
    def test_square_hull(self):
        """Points clearly inside / outside a square hull are classified."""
        data_x = np.array([0.0, 1.0, 1.0, 0.0])
        data_y = np.array([0.0, 0.0, 1.0, 1.0])
        qx = np.array([0.5, -0.5, 1.5, 0.5, 0.5])
        qy = np.array([0.5, 0.5, 0.5, -0.5, 1.5])
        inside = convex_hull_mask(data_x, data_y, qx, qy)
        np.testing.assert_array_equal(
            inside, [True, False, False, False, False],
        )

    def test_preserves_shape(self):
        """Mask should have the same shape as the query array."""
        data_x = np.random.RandomState(0).uniform(-1, 1, 50)
        data_y = np.random.RandomState(1).uniform(-1, 1, 50)
        Xq, Yq = np.meshgrid(np.linspace(-2, 2, 7), np.linspace(-2, 2, 5))
        inside = convex_hull_mask(data_x, data_y, Xq, Yq)
        assert inside.shape == Xq.shape
        assert inside.dtype == bool
