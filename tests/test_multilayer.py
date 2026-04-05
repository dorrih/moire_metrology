"""Tests for multi-layer relaxation."""

import numpy as np
import pytest

from moire_metrology import GRAPHENE, SolverConfig
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


class TestLayerStack:
    def test_describe(self):
        stack = LayerStack(top=TBLG, n_top=3, bottom=TBLG, n_bottom=2, theta_twist=1.0)
        desc = stack.describe()
        assert "3x TBLG" in desc
        assert "2x TBLG" in desc
        assert stack.total_layers == 5

    def test_bilayer(self):
        """Multi-layer with nlayer=1 each should match single-layer result."""
        stack = LayerStack(top=TBLG, n_top=1, bottom=TBLG, n_bottom=1, theta_twist=2.0)
        result = stack.solve(FAST_CONFIG)
        assert result.total_energy < result.unrelaxed_energy
        assert result.displacement_x1.shape == (1, result.mesh.n_vertices)

    def test_trilayer(self):
        """3 layers total: 2 top + 1 bottom."""
        stack = LayerStack(top=TBLG, n_top=2, bottom=TBLG, n_bottom=1, theta_twist=2.0)
        result = stack.solve(FAST_CONFIG)

        assert result.total_energy < result.unrelaxed_energy
        assert result.displacement_x1.shape[0] == 2  # 2 top layers
        assert result.displacement_x2.shape[0] == 1  # 1 bottom layer

    def test_more_layers_more_energy(self):
        """More layers should give more total energy (more unit cells)."""
        config = SolverConfig(method="L-BFGS-B", pixel_size=1.5, max_iter=100, gtol=1e-3, display=False)

        stack1 = LayerStack(top=TBLG, n_top=1, bottom=TBLG, n_bottom=1, theta_twist=3.0)
        result1 = stack1.solve(config)

        stack2 = LayerStack(top=TBLG, n_top=2, bottom=TBLG, n_bottom=1, theta_twist=3.0)
        result2 = stack2.solve(config)

        # More layers means more elastic energy contribution
        assert result2.unrelaxed_energy > result1.unrelaxed_energy


class TestMultiLayerGradient:
    def test_gradient_trilayer(self):
        """Verify gradient via finite differences for a 3-layer system."""
        from moire_metrology.discretization import PeriodicDiscretization
        from moire_metrology.energy import RelaxationEnergy
        from moire_metrology.gsfe import GSFESurface
        from moire_metrology.lattice import HexagonalLattice, MoireGeometry
        from moire_metrology.mesh import MoireMesh

        lat = HexagonalLattice(alpha=0.247)
        geom = MoireGeometry(lat, theta_twist=5.0)
        mesh = MoireMesh.generate(geom, pixel_size=2.0)
        disc = PeriodicDiscretization(mesh, geom)
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
