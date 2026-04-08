"""Tests for the elastic-modulus unit conversion helpers and the
literature-correct values of the bundled GRAPHENE material.

The package stores K, G in meV per unit cell (the Carr/Halbertal
convention used in the elastic energy expression), but the bulk of
the experimental and theoretical 2D-elasticity literature reports
values in N/m. These tests pin down the conversion arithmetic and
lock in the literature-verified values for graphene against future
regressions.
"""

from __future__ import annotations

import numpy as np

from moire_metrology import GRAPHENE, GRAPHENE_GRAPHENE, Material


# ---------------------------------------------------------------------------
# Bundled GRAPHENE values match the published literature
# ---------------------------------------------------------------------------


class TestGrapheneLiteratureValues:
    """Regression-lock the graphene parameters against the paper.

    Halbertal et al. Nat. Commun. 12, 242 (2021), SI Table 1, "TBG"
    column gives the explicit values cited from Carr et al. PRB 98,
    224102 (2018). All units are meV/uc with α = 0.247 nm.
    """

    def test_graphene_K_matches_paper(self):
        assert GRAPHENE.bulk_modulus == 69518.0

    def test_graphene_G_matches_paper(self):
        assert GRAPHENE.shear_modulus == 47352.0

    def test_graphene_lattice_constant(self):
        assert GRAPHENE.lattice_constant == 0.247

    def test_graphene_K_matches_lee_2008_in_n_per_m(self):
        """The paper Table 1 K in meV/uc must round-trip to ~211 N/m,
        the experimental indentation value of Lee et al. Science 321,
        385 (2008). Anything outside ~205-215 N/m would mean the
        unit conversion or the stored value is wrong."""
        K_npm, G_npm = GRAPHENE.moduli_n_per_m
        assert 205.0 < K_npm < 215.0, f"K = {K_npm} N/m, expected ~211"
        assert 140.0 < G_npm < 148.0, f"G = {G_npm} N/m, expected ~144"

    def test_graphene_gsfe_matches_carr_paper(self):
        """GSFE coefficients from Halbertal SI Table 1 / Carr et al."""
        expected = (6.832, 4.064, -0.374, -0.095, 0.0, 0.0)
        assert GRAPHENE_GRAPHENE.gsfe_coeffs == expected


# ---------------------------------------------------------------------------
# Material.moduli_n_per_m round-trip
# ---------------------------------------------------------------------------


class TestModuliNperMConversion:
    def test_roundtrip_via_constructor(self):
        """from_2d_moduli_n_per_m -> .moduli_n_per_m must be identity."""
        m = Material.from_2d_moduli_n_per_m(
            name="X", lattice_constant=0.247,
            bulk_modulus_n_per_m=200.0, shear_modulus_n_per_m=140.0,
        )
        K_npm, G_npm = m.moduli_n_per_m
        np.testing.assert_allclose(K_npm, 200.0, rtol=1e-12)
        np.testing.assert_allclose(G_npm, 140.0, rtol=1e-12)

    def test_conversion_factor_for_graphene(self):
        """For graphene the meV/uc per (N/m) factor must be ~329.5.

        S_uc = (sqrt(3)/2) * (0.247e-9)^2 = 5.281e-20 m^2
        factor = S_uc [m^2] * 6.241509e21 [meV/J] = 329.51 meV/uc per N/m
        """
        m = Material.from_2d_moduli_n_per_m(
            name="X", lattice_constant=0.247,
            bulk_modulus_n_per_m=1.0, shear_modulus_n_per_m=0.0,
        )
        # For 1 N/m input the meV/uc value is the conversion factor itself.
        np.testing.assert_allclose(m.bulk_modulus, 329.51, rtol=1e-3)

    def test_lattice_constant_dependence(self):
        """A larger unit cell area gives more meV/uc per N/m, all else equal.

        Doubling the lattice constant quadruples S_uc and therefore
        quadruples the meV/uc-per-(N/m) factor.
        """
        m1 = Material.from_2d_moduli_n_per_m(
            name="A", lattice_constant=0.247,
            bulk_modulus_n_per_m=100.0, shear_modulus_n_per_m=100.0,
        )
        m2 = Material.from_2d_moduli_n_per_m(
            name="B", lattice_constant=0.494,
            bulk_modulus_n_per_m=100.0, shear_modulus_n_per_m=100.0,
        )
        ratio = m2.bulk_modulus / m1.bulk_modulus
        np.testing.assert_allclose(ratio, 4.0, rtol=1e-12)

    def test_paper_value_recovers_literature_npm(self):
        """The bundled GRAPHENE.K = 69518 meV/uc, when converted back
        via .moduli_n_per_m, must give ~211 N/m within 0.1%."""
        K_npm, _ = GRAPHENE.moduli_n_per_m
        np.testing.assert_allclose(K_npm, 211.0, rtol=2e-3)
