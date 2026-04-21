"""Tests for the n-layer stacking helper and the bundled TDBG interface."""
import pytest

from moire_metrology import (
    GRAPHENE,
    GRAPHENE_BILAYER,
    GRAPHENE_BILAYER_GRAPHENE_BILAYER,
    Material,
)


class TestNLayerStack:
    def test_n1_reproduces_base_moduli(self):
        m = Material.n_layer_stack(GRAPHENE, n=1)
        assert m.bulk_modulus == GRAPHENE.bulk_modulus
        assert m.shear_modulus == GRAPHENE.shear_modulus
        assert m.lattice_constant == GRAPHENE.lattice_constant

    def test_n2_scales_moduli_linearly(self):
        m = Material.n_layer_stack(GRAPHENE, n=2)
        assert m.bulk_modulus == 2 * GRAPHENE.bulk_modulus
        assert m.shear_modulus == 2 * GRAPHENE.shear_modulus

    def test_n3_scales_moduli_linearly(self):
        m = Material.n_layer_stack(GRAPHENE, n=3)
        assert m.bulk_modulus == 3 * GRAPHENE.bulk_modulus
        assert m.shear_modulus == 3 * GRAPHENE.shear_modulus

    def test_lattice_constant_is_inherited(self):
        m = Material.n_layer_stack(GRAPHENE, n=5)
        assert m.lattice_constant == GRAPHENE.lattice_constant

    def test_default_name_carries_n(self):
        m = Material.n_layer_stack(GRAPHENE, n=2)
        assert "n=2" in m.name
        assert GRAPHENE.name in m.name

    def test_custom_name_overrides_default(self):
        m = Material.n_layer_stack(GRAPHENE, n=2, name="Bernal bilayer")
        assert m.name == "Bernal bilayer"

    def test_n_defaults_to_one(self):
        m = Material.n_layer_stack(GRAPHENE)
        assert m.bulk_modulus == GRAPHENE.bulk_modulus

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            Material.n_layer_stack(GRAPHENE, n=0)
        with pytest.raises(ValueError, match="n must be >= 1"):
            Material.n_layer_stack(GRAPHENE, n=-1)


class TestGrapheneBilayer:
    def test_moduli_are_twice_graphene(self):
        assert GRAPHENE_BILAYER.bulk_modulus == 2 * GRAPHENE.bulk_modulus
        assert GRAPHENE_BILAYER.shear_modulus == 2 * GRAPHENE.shear_modulus

    def test_matches_halbertal_table1_values(self):
        # SI Table 1 reports K = 139036, G = 94704 meV/uc for TDBG,
        # which is 2 * the TBG row (69518 / 47352). The bundled value
        # must reproduce that to avoid a subtle regression.
        assert GRAPHENE_BILAYER.bulk_modulus == pytest.approx(139036.0)
        assert GRAPHENE_BILAYER.shear_modulus == pytest.approx(94704.0)

    def test_lattice_constant_matches_graphene(self):
        assert GRAPHENE_BILAYER.lattice_constant == GRAPHENE.lattice_constant


class TestTdbgInterface:
    def test_is_homobilayer(self):
        assert GRAPHENE_BILAYER_GRAPHENE_BILAYER.is_homobilayer

    def test_uses_graphene_bilayer_on_both_sides(self):
        assert GRAPHENE_BILAYER_GRAPHENE_BILAYER.bottom is GRAPHENE_BILAYER
        assert GRAPHENE_BILAYER_GRAPHENE_BILAYER.top is GRAPHENE_BILAYER

    def test_gsfe_matches_dft_d2_row(self):
        # Halbertal et al. Nat. Commun. 12, 242 (2021), SI Table 1,
        # TDBG (DFT-D2) column, c0..c5 verbatim.
        expected = (10.4395, 6.0761, -0.4995, -0.1972, 0.0453, 0.0019)
        assert GRAPHENE_BILAYER_GRAPHENE_BILAYER.gsfe_coeffs == expected

    def test_reference_mentions_dft_d2_and_halbertal(self):
        ref = GRAPHENE_BILAYER_GRAPHENE_BILAYER.reference
        assert "DFT-D2" in ref
        assert "Halbertal" in ref
