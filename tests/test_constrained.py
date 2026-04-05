"""Tests for constrained relaxation with pinned stacking sites."""

import numpy as np
import pytest

from moire_metrology.discretization import PeriodicDiscretization, PinnedConstraints
from moire_metrology.energy import RelaxationEnergy
from moire_metrology.gsfe import GSFESurface
from moire_metrology.lattice import HexagonalLattice, MoireGeometry
from moire_metrology.materials import Material
from moire_metrology.mesh import MoireMesh, generate_finite_mesh
from moire_metrology.pinning import PinningMap, STACKING_PHASES
from moire_metrology.solver import RelaxationSolver, SolverConfig


TBLG = Material(
    name="TBLG",
    lattice_constant=0.247,
    bulk_modulus=69518.0,
    shear_modulus=47352.0,
    gsfe_coeffs=(6.832, 4.064, -0.374, -0.095, 0.0, 0.0),
)


@pytest.fixture
def setup_periodic():
    """Set up a periodic mesh + discretization for testing constraints."""
    lat = HexagonalLattice(alpha=0.247)
    geom = MoireGeometry(lat, theta_twist=2.0)
    mesh = MoireMesh.generate(geom, pixel_size=1.0)
    disc = PeriodicDiscretization(mesh, geom)
    conv = disc.build_conversion_matrices(nlayer1=1, nlayer2=1)
    gsfe = GSFESurface(TBLG.gsfe_coeffs)
    return mesh, geom, disc, conv, gsfe


class TestPinnedConstraints:
    def test_expand_project_roundtrip(self):
        """Expand then project should recover the free DOFs."""
        n_full = 100
        free_idx = np.array([0, 1, 3, 5, 7, 10, 20, 50, 99])
        pinned_idx = np.setdiff1d(np.arange(n_full), free_idx)
        pinned_vals = np.random.randn(len(pinned_idx))

        c = PinnedConstraints(
            free_indices=free_idx, pinned_indices=pinned_idx,
            pinned_values=pinned_vals, n_free=len(free_idx), n_full=n_full,
        )

        U_free = np.random.randn(c.n_free)
        U_full = c.expand(U_free)
        U_free_back = c.project(U_full)
        np.testing.assert_allclose(U_free_back, U_free)

    def test_pinned_values_in_expanded(self):
        """Expanded vector should have correct pinned values."""
        n_full = 20
        pinned_idx = np.array([2, 5, 10])
        pinned_vals = np.array([1.0, 2.0, 3.0])
        free_idx = np.setdiff1d(np.arange(n_full), pinned_idx)

        c = PinnedConstraints(
            free_indices=free_idx, pinned_indices=pinned_idx,
            pinned_values=pinned_vals, n_free=len(free_idx), n_full=n_full,
        )

        U_free = np.zeros(c.n_free)
        U_full = c.expand(U_free)

        np.testing.assert_allclose(U_full[2], 1.0)
        np.testing.assert_allclose(U_full[5], 2.0)
        np.testing.assert_allclose(U_full[10], 3.0)


class TestPinningMap:
    def test_pin_stacking_finds_vertices(self, setup_periodic):
        mesh, geom, disc, conv, gsfe = setup_periodic
        pins = PinningMap(mesh, geom)

        # Pin at the center of the domain
        cx = np.mean(mesh.points[0])
        cy = np.mean(mesh.points[1])
        pins.pin_stacking(x=cx, y=cy, stacking="AB", radius=1.0)

        pinned = pins.get_pinned_vertex_indices()
        assert len(pinned) > 0

    def test_build_constraints(self, setup_periodic):
        mesh, geom, disc, conv, gsfe = setup_periodic
        pins = PinningMap(mesh, geom)

        cx = np.mean(mesh.points[0])
        cy = np.mean(mesh.points[1])
        pins.pin_stacking(x=cx, y=cy, stacking="AB", radius=2.0)

        constraints = pins.build_constraints(conv, nlayer1=1, nlayer2=1)
        assert constraints.n_free < constraints.n_full
        assert constraints.n_free + len(constraints.pinned_indices) == constraints.n_full


class TestConstrainedEnergy:
    def test_gradient_with_constraints(self, setup_periodic):
        """Verify gradient with constraints via finite differences."""
        mesh, geom, disc, conv, gsfe = setup_periodic

        pins = PinningMap(mesh, geom)
        cx = np.mean(mesh.points[0])
        cy = np.mean(mesh.points[1])
        pins.pin_stacking(x=cx, y=cy, stacking="AB", radius=2.0)
        constraints = pins.build_constraints(conv)

        energy = RelaxationEnergy(
            disc=disc, conv=conv, geometry=geom, gsfe_interface=gsfe,
            K1=TBLG.bulk_modulus, G1=TBLG.shear_modulus,
            K2=TBLG.bulk_modulus, G2=TBLG.shear_modulus,
            constraints=constraints,
        )

        np.random.seed(42)
        U_free = np.random.randn(constraints.n_free) * 0.001

        E, grad = energy(U_free)
        assert len(grad) == constraints.n_free

        # Finite difference check
        h = 1e-6
        n_check = min(15, constraints.n_free)
        indices = np.random.choice(constraints.n_free, n_check, replace=False)

        for idx in indices:
            U_plus = U_free.copy()
            U_plus[idx] += h
            E_plus, _ = energy(U_plus)
            U_minus = U_free.copy()
            U_minus[idx] -= h
            E_minus, _ = energy(U_minus)

            grad_fd = (E_plus - E_minus) / (2 * h)
            np.testing.assert_allclose(
                grad[idx], grad_fd, rtol=1e-3, atol=1e-8,
                err_msg=f"Constrained gradient mismatch at free index {idx}"
            )

    def test_hessian_with_constraints(self, setup_periodic):
        """Verify Hessian with constraints via finite differences of gradient."""
        mesh, geom, disc, conv, gsfe = setup_periodic

        pins = PinningMap(mesh, geom)
        cx = np.mean(mesh.points[0])
        cy = np.mean(mesh.points[1])
        pins.pin_stacking(x=cx, y=cy, stacking="AB", radius=2.0)
        constraints = pins.build_constraints(conv)

        energy = RelaxationEnergy(
            disc=disc, conv=conv, geometry=geom, gsfe_interface=gsfe,
            K1=TBLG.bulk_modulus, G1=TBLG.shear_modulus,
            K2=TBLG.bulk_modulus, G2=TBLG.shear_modulus,
            constraints=constraints,
        )

        np.random.seed(99)
        U_free = np.random.randn(constraints.n_free) * 0.001

        h = 1e-6
        n_check = 10
        indices = np.random.choice(constraints.n_free, n_check, replace=False)

        for idx in indices:
            U_plus = U_free.copy()
            U_plus[idx] += h
            _, grad_plus = energy(U_plus)
            U_minus = U_free.copy()
            U_minus[idx] -= h
            _, grad_minus = energy(U_minus)
            hess_col_fd = (grad_plus - grad_minus) / (2 * h)

            e_i = np.zeros(constraints.n_free)
            e_i[idx] = 1.0
            hess_col = energy.hessp(U_free, e_i)

            np.testing.assert_allclose(
                hess_col, hess_col_fd, rtol=1e-3, atol=1e-6,
                err_msg=f"Constrained Hessian mismatch at column {idx}"
            )


class TestConstrainedSolver:
    def test_solve_with_pins(self):
        """Constrained solve should converge with pinned sites."""
        lat = HexagonalLattice(alpha=0.247)
        geom = MoireGeometry(lat, theta_twist=2.0)
        mesh = MoireMesh.generate(geom, pixel_size=1.0)
        disc = PeriodicDiscretization(mesh, geom)
        conv = disc.build_conversion_matrices()

        pins = PinningMap(mesh, geom)
        # Pin a region to AB stacking
        cx, cy = np.mean(mesh.points[0]), np.mean(mesh.points[1])
        pins.pin_stacking(x=cx, y=cy, stacking="AB", radius=1.5)
        constraints = pins.build_constraints(conv)

        config = SolverConfig(method="L-BFGS-B", pixel_size=1.0, max_iter=100,
                              gtol=1e-4, display=False)
        solver = RelaxationSolver(config)
        result = solver.solve(
            material1=TBLG, material2=TBLG, theta_twist=2.0,
            constraints=constraints,
        )

        assert result.total_energy < result.unrelaxed_energy


class TestFiniteMesh:
    def test_generate_finite(self):
        lat = HexagonalLattice(alpha=0.247)
        geom = MoireGeometry(lat, theta_twist=2.0)
        mesh = generate_finite_mesh(geom, n_cells=2, pixel_size=1.0)
        assert mesh.n_vertices > 0
        assert mesh.n_triangles > 0
        # Non-periodic: all triangle indices should be within bounds
        assert np.all(mesh.triangles < mesh.n_vertices)
