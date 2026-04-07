"""Tests for multi-layer relaxation."""

import numpy as np
import pytest

from moire_metrology import SolverConfig
from moire_metrology.materials import Material
from moire_metrology.multilayer import LayerStack


TBLG = Material(
    name="TBLG",
    lattice_constant=0.247,
    bulk_modulus=69518.0,
    shear_modulus=47352.0,
    gsfe_coeffs=(6.832, 4.064, -0.374, -0.095, 0.0, 0.0),
    stacking_func=lambda k: ((1 / 3) * ((-1) ** k), (1 / 3) * ((-1) ** k)),
)

FAST_CONFIG = SolverConfig(method="L-BFGS-B", pixel_size=1.0, max_iter=200, gtol=1e-4, display=False)

# Even smaller config for fix_top/fix_bottom tests — runs in <2s.
TINY_CONFIG = SolverConfig(method="L-BFGS-B", pixel_size=1.5, max_iter=100,
                           gtol=1e-3, display=False, min_mesh_points=20)


class TestLayerStack:
    def test_describe(self):
        stack = LayerStack(top=TBLG, n_top=3, bottom=TBLG, n_bottom=2, theta_twist=1.0)
        desc = stack.describe()
        assert "3x TBLG" in desc
        assert "2x TBLG" in desc
        assert stack.total_layers == 5

    @pytest.mark.slow
    def test_bilayer(self):
        """Multi-layer with nlayer=1 each should match single-layer result."""
        stack = LayerStack(top=TBLG, n_top=1, bottom=TBLG, n_bottom=1, theta_twist=2.0)
        result = stack.solve(FAST_CONFIG)
        assert result.total_energy < result.unrelaxed_energy
        assert result.displacement_x1.shape == (1, result.mesh.n_vertices)

    @pytest.mark.slow
    def test_trilayer(self):
        """3 layers total: 2 top + 1 bottom."""
        stack = LayerStack(top=TBLG, n_top=2, bottom=TBLG, n_bottom=1, theta_twist=2.0)
        result = stack.solve(FAST_CONFIG)

        assert result.total_energy < result.unrelaxed_energy
        assert result.displacement_x1.shape[0] == 2  # 2 top layers
        assert result.displacement_x2.shape[0] == 1  # 1 bottom layer

    @pytest.mark.slow
    def test_more_layers_more_energy(self):
        """More layers should give more total energy (more unit cells)."""
        config = SolverConfig(method="L-BFGS-B", pixel_size=1.5, max_iter=100, gtol=1e-3, display=False)

        stack1 = LayerStack(top=TBLG, n_top=1, bottom=TBLG, n_bottom=1, theta_twist=3.0)
        result1 = stack1.solve(config)

        stack2 = LayerStack(top=TBLG, n_top=2, bottom=TBLG, n_bottom=1, theta_twist=3.0)
        result2 = stack2.solve(config)

        # More layers means more elastic energy contribution
        assert result2.unrelaxed_energy > result1.unrelaxed_energy


class TestFixOuterLayers:
    """Tests for the fix_top / fix_bottom layer-clamping options."""

    def test_fix_bottom_pins_deepest_substrate_layer(self):
        """fix_bottom=True must pin all DOFs of the bottommost layer to zero."""
        stack = LayerStack(
            top=TBLG, n_top=1, bottom=TBLG, n_bottom=3, theta_twist=2.0,
        )
        result = stack.solve(TINY_CONFIG, fix_bottom=True)

        # Bottommost (deepest) substrate layer is index n_bottom - 1
        deepest_ux = result.displacement_x2[-1]
        deepest_uy = result.displacement_y2[-1]
        assert np.allclose(deepest_ux, 0.0, atol=1e-12), \
            f"deepest layer ux should be zero, got max abs {np.abs(deepest_ux).max()}"
        assert np.allclose(deepest_uy, 0.0, atol=1e-12)

        # Free layers (above) should be allowed to relax
        free_max = max(np.abs(result.displacement_x2[k]).max() for k in range(2))
        assert free_max > 0.0

        # Constrained relaxation still reduces energy from the rigid state
        assert result.total_energy < result.unrelaxed_energy

    def test_fix_top_pins_topmost_layer(self):
        """fix_top=True must pin all DOFs of the topmost layer to zero."""
        stack = LayerStack(
            top=TBLG, n_top=2, bottom=TBLG, n_bottom=1, theta_twist=2.0,
        )
        result = stack.solve(TINY_CONFIG, fix_top=True)

        # Topmost layer is stack 1, layer 0
        top_ux = result.displacement_x1[0]
        top_uy = result.displacement_y1[0]
        assert np.allclose(top_ux, 0.0, atol=1e-12)
        assert np.allclose(top_uy, 0.0, atol=1e-12)

        # The other top-flake layer (layer 1, adjacent to the interface)
        # should be free to relax.
        inner_max = float(np.abs(result.displacement_x1[1]).max())
        assert inner_max > 0.0

    def test_fix_both_clamps_both_outer_layers(self):
        stack = LayerStack(
            top=TBLG, n_top=2, bottom=TBLG, n_bottom=2, theta_twist=2.0,
        )
        result = stack.solve(TINY_CONFIG, fix_top=True, fix_bottom=True)

        assert np.allclose(result.displacement_x1[0], 0.0, atol=1e-12)
        assert np.allclose(result.displacement_y1[0], 0.0, atol=1e-12)
        assert np.allclose(result.displacement_x2[-1], 0.0, atol=1e-12)
        assert np.allclose(result.displacement_y2[-1], 0.0, atol=1e-12)

    def test_constrained_energy_above_unconstrained(self):
        """Clamping the bottom layer must give a (weakly) higher relaxed energy
        than the unconstrained case — pinning is a strict reduction in
        feasible set."""
        stack = LayerStack(
            top=TBLG, n_top=1, bottom=TBLG, n_bottom=3, theta_twist=2.0,
        )
        free = stack.solve(TINY_CONFIG)
        clamped = stack.solve(TINY_CONFIG, fix_bottom=True)

        # Both should beat the rigid configuration
        assert free.total_energy < free.unrelaxed_energy
        assert clamped.total_energy < clamped.unrelaxed_energy
        # Clamped is at least as costly as free (with a tiny tolerance for
        # solver noise on tiny meshes)
        assert clamped.total_energy >= free.total_energy - 1e-3 * abs(free.total_energy)

    def test_explicit_constraints_conflict_raises(self):
        """Combining fix_bottom with an explicit constraints object is rejected."""
        from moire_metrology.discretization import (
            Discretization, build_outer_layer_constraints,
        )
        from moire_metrology.lattice import HexagonalLattice, MoireGeometry
        from moire_metrology.mesh import MoireMesh
        from moire_metrology.solver import RelaxationSolver

        lat = HexagonalLattice(alpha=TBLG.lattice_constant)
        geom = MoireGeometry(lat, theta_twist=2.0)
        mesh = MoireMesh.generate(geom, pixel_size=1.5, min_points=20)
        disc = Discretization(mesh, geom)
        conv = disc.build_conversion_matrices(nlayer1=1, nlayer2=2)
        explicit = build_outer_layer_constraints(conv, fix_top=False, fix_bottom=True)

        solver = RelaxationSolver(TINY_CONFIG)
        with pytest.raises(ValueError, match="cannot be combined"):
            solver.solve(
                material1=TBLG, material2=TBLG, theta_twist=2.0,
                nlayer1=1, nlayer2=2,
                constraints=explicit, fix_bottom=True,
            )


class TestMultiLayerGradient:
    def test_gradient_trilayer(self):
        """Verify gradient via finite differences for a 3-layer system."""
        from moire_metrology.discretization import Discretization
        from moire_metrology.energy import RelaxationEnergy
        from moire_metrology.gsfe import GSFESurface
        from moire_metrology.lattice import HexagonalLattice, MoireGeometry
        from moire_metrology.mesh import MoireMesh

        lat = HexagonalLattice(alpha=0.247)
        geom = MoireGeometry(lat, theta_twist=5.0)
        mesh = MoireMesh.generate(geom, pixel_size=2.0)
        disc = Discretization(mesh, geom)
        conv = disc.build_conversion_matrices(nlayer1=2, nlayer2=1)
        gsfe = GSFESurface(TBLG.gsfe_coeffs)

        energy = RelaxationEnergy(
            disc=disc, conv=conv, geometry=geom,
            gsfe_interface=gsfe,
            K1=TBLG.bulk_modulus, G1=TBLG.shear_modulus,
            K2=TBLG.bulk_modulus, G2=TBLG.shear_modulus,
            nlayer1=2, nlayer2=1,
            gsfe_flake1=gsfe,
            I1_vect=np.array([1 / 3]),
            J1_vect=np.array([1 / 3]),
        )

        np.random.seed(42)
        U = np.random.randn(conv.n_sol) * 0.001

        E, grad = energy(U)

        # Finite difference check
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

    def test_hessian_trilayer(self):
        """Verify Hessian via finite differences of gradient for a 3-layer system."""
        from moire_metrology.discretization import Discretization
        from moire_metrology.energy import RelaxationEnergy
        from moire_metrology.gsfe import GSFESurface
        from moire_metrology.lattice import HexagonalLattice, MoireGeometry
        from moire_metrology.mesh import MoireMesh

        lat = HexagonalLattice(alpha=0.247)
        geom = MoireGeometry(lat, theta_twist=5.0)
        mesh = MoireMesh.generate(geom, pixel_size=2.0)
        disc = Discretization(mesh, geom)
        conv = disc.build_conversion_matrices(nlayer1=2, nlayer2=1)
        gsfe = GSFESurface(TBLG.gsfe_coeffs)

        energy = RelaxationEnergy(
            disc=disc, conv=conv, geometry=geom,
            gsfe_interface=gsfe,
            K1=TBLG.bulk_modulus, G1=TBLG.shear_modulus,
            K2=TBLG.bulk_modulus, G2=TBLG.shear_modulus,
            nlayer1=2, nlayer2=1,
            gsfe_flake1=gsfe,
            I1_vect=np.array([1 / 3]),
            J1_vect=np.array([1 / 3]),
        )

        np.random.seed(123)
        U = np.random.randn(conv.n_sol) * 0.001

        # Test hessp against finite differences of gradient
        h = 1e-6
        n_check = 15
        indices = np.random.choice(len(U), n_check, replace=False)

        for idx in indices:
            # Finite difference of gradient: H[:,idx] ~ (grad(U+h*e_i) - grad(U-h*e_i)) / 2h
            U_plus = U.copy()
            U_plus[idx] += h
            _, grad_plus = energy(U_plus)

            U_minus = U.copy()
            U_minus[idx] -= h
            _, grad_minus = energy(U_minus)

            hess_col_fd = (grad_plus - grad_minus) / (2 * h)

            # hessp should give the same column when applied to e_i
            e_i = np.zeros_like(U)
            e_i[idx] = 1.0
            hess_col_hessp = energy.hessp(U, e_i)

            np.testing.assert_allclose(
                hess_col_hessp, hess_col_fd, rtol=1e-3, atol=1e-6,
                err_msg=f"Hessp mismatch at column {idx}"
            )

        # Also verify full sparse Hessian against hessp
        H = energy.hessian(U)
        for idx in indices[:5]:
            e_i = np.zeros_like(U)
            e_i[idx] = 1.0
            col_sparse = H @ e_i
            col_hessp = energy.hessp(U, e_i)
            np.testing.assert_allclose(
                col_sparse, col_hessp, rtol=1e-10, atol=1e-12,
                err_msg=f"Hessian vs hessp mismatch at column {idx}"
            )
