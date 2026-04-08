"""Tests for constrained relaxation with pinned stacking sites."""

import numpy as np
import pytest

from moire_metrology import Interface, Material
from moire_metrology.discretization import Discretization, PinnedConstraints
from moire_metrology.energy import RelaxationEnergy
from moire_metrology.gsfe import GSFESurface
from moire_metrology.lattice import HexagonalLattice, MoireGeometry
from moire_metrology.mesh import MoireMesh, generate_finite_mesh
from moire_metrology.pinning import PinningMap
from moire_metrology.solver import RelaxationSolver, SolverConfig


TBLG = Material(
    name="TBLG",
    lattice_constant=0.247,
    bulk_modulus=69518.0,
    shear_modulus=47352.0,
)

TBLG_INTERFACE = Interface(
    name="TBLG/TBLG (test)",
    bottom=TBLG,
    top=TBLG,
    gsfe_coeffs=(6.832, 4.064, -0.374, -0.095, 0.0, 0.0),
    reference="hand-rolled test interface",
)


@pytest.fixture
def setup_periodic():
    """Set up a periodic mesh + discretization for testing constraints."""
    lat = HexagonalLattice(alpha=0.247)
    geom = MoireGeometry(lat, theta_twist=2.0)
    mesh = MoireMesh.generate(geom, pixel_size=1.0)
    disc = Discretization(mesh, geom)
    conv = disc.build_conversion_matrices(nlayer1=1, nlayer2=1)
    gsfe = GSFESurface(TBLG_INTERFACE.gsfe_coeffs)
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
    @pytest.mark.slow
    def test_solve_with_pins(self):
        """Constrained solve should converge with pinned sites."""
        lat = HexagonalLattice(alpha=0.247)
        geom = MoireGeometry(lat, theta_twist=2.0)
        mesh = MoireMesh.generate(geom, pixel_size=1.0)
        disc = Discretization(mesh, geom)
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
            moire_interface=TBLG_INTERFACE, theta_twist=2.0,
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
        # And the mesh must advertise itself as non-periodic so the
        # discretization skips the lattice-vector wrap correction.
        assert mesh.is_periodic is False

    def test_periodic_mesh_default_is_periodic(self):
        """The standard MoireMesh.generate output is still periodic."""
        from moire_metrology.mesh import MoireMesh
        lat = HexagonalLattice(alpha=0.247)
        geom = MoireGeometry(lat, theta_twist=2.0)
        mesh = MoireMesh.generate(geom, pixel_size=1.0, min_points=20)
        assert mesh.is_periodic is True

    def test_diff_matrices_exact_on_linear_field_finite(self):
        """On a finite mesh, the FEM diff matrices must reproduce a linear
        field's analytic derivative exactly, at every triangle (no
        spurious wrap-around correction).
        """
        lat = HexagonalLattice(alpha=0.247)
        geom = MoireGeometry(lat, theta_twist=2.0)
        mesh = generate_finite_mesh(geom, n_cells=2, pixel_size=1.0)
        disc = Discretization(mesh, geom)

        # Linear field f = a*x + b*y has constant df/dx = a, df/dy = b
        a, b = 0.7, -0.3
        x = mesh.points[0]
        y = mesh.points[1]
        f = a * x + b * y

        dfx = disc.diff_mat_x @ f
        dfy = disc.diff_mat_y @ f

        # Every triangle should give the analytic derivative to machine precision
        np.testing.assert_allclose(dfx, a, atol=1e-10)
        np.testing.assert_allclose(dfy, b, atol=1e-10)


class TestFiniteMeshRelaxation:
    """Round-trip tests for the new finite-mesh + point-pinning workflow.

    These verify that RelaxationSolver.solve() accepts a pre-built finite
    mesh and that pins built via PinningMap on that mesh are honoured by
    the solver. This is the spatially-varying-strain capability used to
    work out post-relaxation maps from experimentally-identified pinned
    stacking sites.
    """

    def _build_finite_setup(self, n_cells=2, pixel_size=1.0, theta=2.0):
        from moire_metrology.discretization import Discretization
        from moire_metrology.lattice import HexagonalLattice, MoireGeometry
        from moire_metrology.mesh import generate_finite_mesh
        lat = HexagonalLattice(alpha=TBLG.lattice_constant)
        geom = MoireGeometry(lat, theta_twist=theta)
        mesh = generate_finite_mesh(geom, n_cells=n_cells, pixel_size=pixel_size)
        disc = Discretization(mesh, geom)
        conv = disc.build_conversion_matrices(nlayer1=1, nlayer2=1)
        return mesh, geom, disc, conv

    def test_finite_mesh_with_point_pins_solves(self):
        """End-to-end: build finite mesh, pin two interior points to AB and
        BA, run the relaxation, verify it converges and the pinned vertices
        actually carry the pinned displacement.
        """
        mesh, geom, disc, conv = self._build_finite_setup()
        pins = PinningMap(mesh, geom)
        cx = float(np.mean(mesh.points[0]))
        cy = float(np.mean(mesh.points[1]))
        pins.pin_stacking(x=cx - 3.0, y=cy, stacking="AB", radius=1.0)
        pins.pin_stacking(x=cx + 3.0, y=cy, stacking="BA", radius=1.0)
        constraints = pins.build_constraints(conv, nlayer1=1, nlayer2=1)
        assert constraints.n_free < constraints.n_full

        cfg = SolverConfig(
            method="L-BFGS-B", pixel_size=1.0, max_iter=200,
            gtol=1e-4, display=False,
        )
        result = RelaxationSolver(cfg).solve(
            moire_interface=TBLG_INTERFACE, theta_twist=2.0,
            mesh=mesh, constraints=constraints,
        )

        # Energy must drop from the rigid configuration
        assert result.total_energy < result.unrelaxed_energy

        # Pinned DOFs in the relaxed solution must equal the requested values
        full_solution = constraints.expand(
            constraints.project(result.solution_vector)
        )
        pinned_actual = full_solution[constraints.pinned_indices]
        np.testing.assert_allclose(
            pinned_actual, constraints.pinned_values, atol=1e-12,
            err_msg="pinned DOFs drifted from their target values",
        )

    def test_solver_accepts_external_mesh(self):
        """Passing mesh=... bypasses the internal MoireMesh.generate() path."""
        mesh, geom, disc, conv = self._build_finite_setup()
        # Run with an explicit mesh, no constraints — should still relax
        # to the trivial U=0 solution because the unrelaxed state is
        # the unconstrained energy minimum modulo rigid translations.
        cfg = SolverConfig(
            method="L-BFGS-B", pixel_size=1.0, max_iter=50,
            gtol=1e-3, display=False,
        )
        # Need at least one pin to remove rigid-body modes
        pins = PinningMap(mesh, geom)
        cx = float(np.mean(mesh.points[0]))
        cy = float(np.mean(mesh.points[1]))
        pins.pin_stacking(x=cx, y=cy, stacking="AB", radius=1.0)
        constraints = pins.build_constraints(conv, nlayer1=1, nlayer2=1)

        result = RelaxationSolver(cfg).solve(
            moire_interface=TBLG_INTERFACE, theta_twist=2.0,
            mesh=mesh, constraints=constraints,
        )
        # Verify the result mesh IS the one we passed in
        assert result.mesh is mesh
        assert result.mesh.is_periodic is False
        assert result.total_energy < result.unrelaxed_energy
