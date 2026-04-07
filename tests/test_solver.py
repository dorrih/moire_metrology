"""Integration tests for the relaxation solver."""

import numpy as np
import pytest

from moire_metrology import RelaxationSolver, SolverConfig, GRAPHENE


class TestPseudoDynamics:
    """Tests for the implicit pseudo-time-stepping solver."""

    def test_matches_newton_on_small_bilayer(self):
        """pseudo_dynamics and newton should agree on energy for a small bilayer."""
        # Small mesh + moderate twist so both solvers run in <5s.
        common = dict(pixel_size=1.5, max_iter=80, gtol=1e-3,
                      display=False, min_mesh_points=30)

        cfg_newton = SolverConfig(method="newton", **common)
        res_newton = RelaxationSolver(cfg_newton).solve(
            material1=GRAPHENE, material2=GRAPHENE, theta_twist=2.0,
        )

        cfg_pd = SolverConfig(method="pseudo_dynamics", **common)
        res_pd = RelaxationSolver(cfg_pd).solve(
            material1=GRAPHENE, material2=GRAPHENE, theta_twist=2.0,
        )

        # Both should reduce energy from the unrelaxed state.
        assert res_newton.total_energy < res_newton.unrelaxed_energy
        assert res_pd.total_energy < res_pd.unrelaxed_energy

        # And they should agree on the relaxed energy to within 1% — they
        # are converging to the same physical minimum, just via different
        # iteration schedules.
        rel_diff = abs(res_pd.total_energy - res_newton.total_energy) / abs(res_newton.total_energy)
        assert rel_diff < 0.01, (
            f"pseudo_dynamics E={res_pd.total_energy:.2f} disagrees with "
            f"newton E={res_newton.total_energy:.2f} (rel diff {rel_diff:.3%})"
        )

    def test_iterative_matches_direct(self):
        """linear_solver='iterative' should give the same answer as 'direct'.

        The iterative path uses matrix-free preconditioned MINRES against
        energy_func.hessp(), while the direct path builds the sparse
        Hessian and calls spsolve. Both should converge to the same
        relaxed energy on a small test case.
        """
        common = dict(method="pseudo_dynamics", pixel_size=1.5, max_iter=80,
                      gtol=1e-3, display=False, min_mesh_points=30)

        cfg_direct = SolverConfig(linear_solver="direct", **common)
        res_direct = RelaxationSolver(cfg_direct).solve(
            material1=GRAPHENE, material2=GRAPHENE, theta_twist=2.0,
        )

        cfg_iter = SolverConfig(linear_solver="iterative", **common)
        res_iter = RelaxationSolver(cfg_iter).solve(
            material1=GRAPHENE, material2=GRAPHENE, theta_twist=2.0,
        )

        # Both should reduce energy from the unrelaxed state.
        assert res_direct.total_energy < res_direct.unrelaxed_energy
        assert res_iter.total_energy < res_iter.unrelaxed_energy

        # And they should agree to high precision — same algorithm, same
        # iteration schedule, only the inner linear solve differs. The
        # MINRES tolerance is 1e-6, so we allow a relative diff of 1e-4
        # to account for accumulated round-off across the outer iterations.
        rel_diff = abs(res_iter.total_energy - res_direct.total_energy) / abs(res_direct.total_energy)
        assert rel_diff < 1e-4, (
            f"iterative E={res_iter.total_energy:.6f} disagrees with "
            f"direct E={res_direct.total_energy:.6f} (rel diff {rel_diff:.3e})"
        )

    def test_iterative_with_constraints(self):
        """The iterative solver path must work with PinnedConstraints too.

        This is a smoke test that the matrix-free hessp + Jacobi
        preconditioner correctly project to the free-DOF space when
        constraints are present.
        """
        cfg = SolverConfig(
            method="pseudo_dynamics", linear_solver="iterative",
            pixel_size=1.5, max_iter=60, gtol=1e-3,
            display=False, min_mesh_points=30,
        )
        # Use fix_bottom on a tiny multilayer — this exercises both the
        # iterative solver and the constraint-aware hessp path.
        from moire_metrology.multilayer import LayerStack
        stack = LayerStack(
            top=GRAPHENE, n_top=1, bottom=GRAPHENE, n_bottom=2, theta_twist=2.0,
        )
        result = stack.solve(cfg, fix_bottom=True)

        # The clamped layer is exactly zero
        assert np.allclose(result.displacement_x2[-1], 0.0, atol=1e-12)
        assert np.allclose(result.displacement_y2[-1], 0.0, atol=1e-12)
        # Free layers relax
        assert result.total_energy < result.unrelaxed_energy
        free_max = float(np.abs(result.displacement_x2[0]).max())
        assert free_max > 0.0

    def test_invalid_linear_solver_raises(self):
        cfg = SolverConfig(
            method="pseudo_dynamics", linear_solver="not_a_thing",
            pixel_size=1.5, max_iter=10, gtol=1e-3,
            display=False, min_mesh_points=30,
        )
        with pytest.raises(ValueError, match="Unknown linear_solver"):
            RelaxationSolver(cfg).solve(
                material1=GRAPHENE, material2=GRAPHENE, theta_twist=2.0,
            )


class TestSolverBasic:
    @pytest.mark.slow
    def test_solve_tblg_2deg(self):
        """Solve TBLG at 2 degrees — energy should decrease from relaxation."""
        config = SolverConfig(
            pixel_size=1.0,  # coarse mesh for speed
            max_iter=200,
            display=False,
        )
        solver = RelaxationSolver(config)
        result = solver.solve(
            material1=GRAPHENE,
            material2=GRAPHENE,
            theta_twist=2.0,
        )

        # Energy should decrease
        assert result.total_energy < result.unrelaxed_energy
        assert result.energy_reduction > 0

        # Displacement fields should exist and have correct shape
        Nv = result.mesh.n_vertices
        assert result.displacement_x1.shape == (1, Nv)
        assert result.displacement_y1.shape == (1, Nv)

    @pytest.mark.slow
    def test_solve_zero_displacement_at_start(self):
        """At large twist angles, relaxation is small."""
        config = SolverConfig(
            pixel_size=1.0,
            max_iter=100,
            display=False,
        )
        solver = RelaxationSolver(config)
        result = solver.solve(
            material1=GRAPHENE,
            material2=GRAPHENE,
            theta_twist=10.0,  # large angle, minimal relaxation
        )

        # At large angles, displacement should be small relative to lattice constant
        max_disp = np.max(np.abs(result.displacement_x1))
        assert max_disp < GRAPHENE.lattice_constant

    @pytest.mark.slow
    def test_gsfe_map_nonnegative(self):
        """GSFE energy density should be non-negative."""
        config = SolverConfig(pixel_size=1.0, max_iter=100, display=False)
        solver = RelaxationSolver(config)
        result = solver.solve(
            material1=GRAPHENE,
            material2=GRAPHENE,
            theta_twist=3.0,
        )
        assert np.all(result.gsfe_map >= -1e-10)


class TestEnergyGradient:
    def test_gradient_finite_difference(self):
        """Verify gradient against finite differences."""
        from moire_metrology.discretization import PeriodicDiscretization
        from moire_metrology.energy import RelaxationEnergy
        from moire_metrology.gsfe import GSFESurface
        from moire_metrology.lattice import HexagonalLattice, MoireGeometry
        from moire_metrology.mesh import MoireMesh

        lat = HexagonalLattice(alpha=0.247)
        geom = MoireGeometry(lat, theta_twist=5.0)
        mesh = MoireMesh.generate(geom, pixel_size=1.5)
        disc = PeriodicDiscretization(mesh, geom)
        conv = disc.build_conversion_matrices(nlayer1=1, nlayer2=1)
        gsfe = GSFESurface(GRAPHENE.gsfe_coeffs)

        energy = RelaxationEnergy(
            disc=disc,
            conv=conv,
            geometry=geom,
            gsfe_interface=gsfe,
            K1=GRAPHENE.bulk_modulus,
            G1=GRAPHENE.shear_modulus,
            K2=GRAPHENE.bulk_modulus,
            G2=GRAPHENE.shear_modulus,
        )

        # Random displacement (small)
        np.random.seed(42)
        U = np.random.randn(conv.n_sol) * 0.001

        E, grad = energy(U)

        # Finite difference check on a subset of DOFs
        h = 1e-6
        n_check = min(20, len(U))
        indices = np.random.choice(len(U), n_check, replace=False)

        for idx in indices:
            U_plus = U.copy()
            U_plus[idx] += h
            E_plus, _ = energy(U_plus)

            U_minus = U.copy()
            U_minus[idx] -= h
            E_minus, _ = energy(U_minus)

            grad_fd = (E_plus - E_minus) / (2 * h)
            np.testing.assert_allclose(
                grad[idx], grad_fd, rtol=1e-3, atol=1e-8,
                err_msg=f"Gradient mismatch at index {idx}"
            )
