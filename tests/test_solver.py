"""Integration tests for the relaxation solver."""

import numpy as np
import pytest

from moire_metrology import RelaxationSolver, SolverConfig, GRAPHENE


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
