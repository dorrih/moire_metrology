"""Tests for the GSFE energy surface."""

import numpy as np
import pytest

from moire_metrology.gsfe import GSFESurface
from moire_metrology.materials import GRAPHENE


@pytest.fixture
def graphene_gsfe():
    return GSFESurface(GRAPHENE.gsfe_coeffs)


class TestGSFEValues:
    """Test GSFE values at high-symmetry stacking points."""

    def test_aa_stacking(self, graphene_gsfe):
        """AA stacking is at (v, w) = (0, 0) and should be an energy maximum."""
        v = np.array(0.0)
        w = np.array(0.0)
        E_aa = graphene_gsfe(v, w)
        # AA should be higher than the minimum (AB)
        assert E_aa > graphene_gsfe.minimum_value

    def test_ab_stacking(self, graphene_gsfe):
        """AB stacking is at (v, w) = (2*pi/3, 2*pi/3) for Carr convention."""
        v_ab = np.array(2 * np.pi / 3)
        w_ab = np.array(2 * np.pi / 3)
        E_ab = graphene_gsfe(v_ab, w_ab)
        np.testing.assert_allclose(E_ab, graphene_gsfe.minimum_value, atol=0.01)

    def test_symmetry(self, graphene_gsfe):
        """For graphene (c4=c5~0 after transform), V(v,w) = V(w,v)."""
        v = np.linspace(0, 2 * np.pi, 50)
        w = np.linspace(0, 2 * np.pi, 50)
        vv, ww = np.meshgrid(v, w)
        V1 = graphene_gsfe(vv, ww)
        V2 = graphene_gsfe(ww, vv)
        # For centrosymmetric materials, V(v,w) ~ V(w,v)
        # This may not be exact due to the AB-reference transform
        # but the difference should be small
        diff = np.max(np.abs(V1 - V2))
        assert diff < 1.0  # should be small for graphene

    def test_periodicity(self, graphene_gsfe):
        """V(v, w) should be 2*pi periodic in both arguments."""
        v = np.linspace(0, 2 * np.pi, 20)
        w = np.linspace(0, 2 * np.pi, 20)
        vv, ww = np.meshgrid(v, w)
        V1 = graphene_gsfe(vv, ww)
        V2 = graphene_gsfe(vv + 2 * np.pi, ww)
        V3 = graphene_gsfe(vv, ww + 2 * np.pi)
        np.testing.assert_allclose(V1, V2, atol=1e-10)
        np.testing.assert_allclose(V1, V3, atol=1e-10)


class TestGSFEDerivatives:
    """Test GSFE derivatives against finite differences."""

    def test_dv_finite_diff(self, graphene_gsfe):
        h = 1e-7
        v = np.linspace(0.1, 6.0, 30)
        w = np.linspace(0.1, 6.0, 30)
        vv, ww = np.meshgrid(v, w)

        analytical = graphene_gsfe.dv(vv, ww)
        numerical = (graphene_gsfe(vv + h, ww) - graphene_gsfe(vv - h, ww)) / (2 * h)
        np.testing.assert_allclose(analytical, numerical, atol=1e-4)

    def test_dw_finite_diff(self, graphene_gsfe):
        h = 1e-7
        v = np.linspace(0.1, 6.0, 30)
        w = np.linspace(0.1, 6.0, 30)
        vv, ww = np.meshgrid(v, w)

        analytical = graphene_gsfe.dw(vv, ww)
        numerical = (graphene_gsfe(vv, ww + h) - graphene_gsfe(vv, ww - h)) / (2 * h)
        np.testing.assert_allclose(analytical, numerical, atol=1e-4)

    def test_d2v2_finite_diff(self, graphene_gsfe):
        h = 1e-5
        v = np.linspace(0.1, 6.0, 20)
        w = np.linspace(0.1, 6.0, 20)
        vv, ww = np.meshgrid(v, w)

        analytical = graphene_gsfe.d2v2(vv, ww)
        numerical = (graphene_gsfe.dv(vv + h, ww) - graphene_gsfe.dv(vv - h, ww)) / (2 * h)
        np.testing.assert_allclose(analytical, numerical, atol=1e-2)

    def test_d2w2_finite_diff(self, graphene_gsfe):
        h = 1e-5
        v = np.linspace(0.1, 6.0, 20)
        w = np.linspace(0.1, 6.0, 20)
        vv, ww = np.meshgrid(v, w)

        analytical = graphene_gsfe.d2w2(vv, ww)
        numerical = (graphene_gsfe.dw(vv, ww + h) - graphene_gsfe.dw(vv, ww - h)) / (2 * h)
        np.testing.assert_allclose(analytical, numerical, atol=1e-2)

    def test_d2vw_finite_diff(self, graphene_gsfe):
        h = 1e-5
        v = np.linspace(0.1, 6.0, 20)
        w = np.linspace(0.1, 6.0, 20)
        vv, ww = np.meshgrid(v, w)

        analytical = graphene_gsfe.d2vw(vv, ww)
        numerical = (graphene_gsfe.dv(vv, ww + h) - graphene_gsfe.dv(vv, ww - h)) / (2 * h)
        np.testing.assert_allclose(analytical, numerical, atol=1e-2)


class TestGSFESaddlePoint:
    def test_saddle_point_positive(self, graphene_gsfe):
        """Saddle point energy should be positive (barrier between AB/BA)."""
        E_sp = graphene_gsfe.saddle_point_energy()
        assert E_sp > 0

    def test_saddle_below_aa(self, graphene_gsfe):
        """Saddle point should be below AA stacking energy."""
        E_sp = graphene_gsfe.saddle_point_energy()
        E_aa = float(graphene_gsfe(np.array(0.0), np.array(0.0))) - graphene_gsfe.minimum_value
        assert E_sp < E_aa
