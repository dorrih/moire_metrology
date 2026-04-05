"""Generalized Stacking Fault Energy (GSFE) surface and its derivatives.

The GSFE is parameterized as a 6-coefficient Fourier expansion in stacking
phase coordinates (v, w):

    V(v,w) = c0 + c1*(cos(v) + cos(w) + cos(v+w))
           + c2*(cos(v+2w) + cos(v-w) + cos(2v+w))
           + c3*(cos(2v) + cos(2w) + cos(2v+2w))
           + c4*(sin(v) + sin(w) - sin(v+w))
           + c5*(sin(2v+2w) - sin(2v) - sin(2w))

All methods operate element-wise on numpy arrays.
"""

from __future__ import annotations

import numpy as np
from numpy import cos, sin
from scipy.optimize import minimize_scalar


class GSFESurface:
    """GSFE energy surface with analytical derivatives.

    Parameters
    ----------
    coeffs : sequence of float
        (c0, c1, c2, c3, c4, c5) in meV/unit cell.
    """

    def __init__(self, coeffs: tuple[float, ...] | list[float]):
        if len(coeffs) != 6:
            raise ValueError(f"Expected 6 GSFE coefficients, got {len(coeffs)}")
        self.c = tuple(float(c) for c in coeffs)
        self._min_val: float | None = None

    def __call__(self, v: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Evaluate V(v, w)."""
        c0, c1, c2, c3, c4, c5 = self.c
        return (
            c0
            + c1 * (cos(v) + cos(w) + cos(v + w))
            + c2 * (cos(v + 2 * w) + cos(v - w) + cos(2 * v + w))
            + c3 * (cos(2 * v) + cos(2 * w) + cos(2 * v + 2 * w))
            + c4 * (sin(v) + sin(w) - sin(v + w))
            + c5 * (sin(2 * v + 2 * w) - sin(2 * v) - sin(2 * w))
        )

    def dv(self, v: np.ndarray, w: np.ndarray) -> np.ndarray:
        """dV/dv."""
        c0, c1, c2, c3, c4, c5 = self.c
        return (
            c1 * (-sin(v) - sin(v + w))
            + c2 * (-sin(v + 2 * w) - sin(v - w) - 2 * sin(2 * v + w))
            + c3 * (-2 * sin(2 * v) - 2 * sin(2 * v + 2 * w))
            + c4 * (cos(v) - cos(v + w))
            + c5 * (2 * cos(2 * v + 2 * w) - 2 * cos(2 * v))
        )

    def dw(self, v: np.ndarray, w: np.ndarray) -> np.ndarray:
        """dV/dw."""
        c0, c1, c2, c3, c4, c5 = self.c
        return (
            c1 * (-sin(w) - sin(v + w))
            + c2 * (-2 * sin(v + 2 * w) + sin(v - w) - sin(2 * v + w))
            + c3 * (-2 * sin(2 * w) - 2 * sin(2 * v + 2 * w))
            + c4 * (cos(w) + cos(v + w))
            + c5 * (2 * cos(2 * v + 2 * w) + 2 * cos(2 * w))
        )

    def d2v2(self, v: np.ndarray, w: np.ndarray) -> np.ndarray:
        """d^2V/dv^2."""
        c0, c1, c2, c3, c4, c5 = self.c
        return (
            c1 * (-cos(v) - cos(v + w))
            + c2 * (-cos(v + 2 * w) - cos(v - w) - 4 * cos(2 * v + w))
            + c3 * (-4 * cos(2 * v) - 4 * cos(2 * v + 2 * w))
            + c4 * (-sin(v) + sin(v + w))
            + c5 * (-4 * sin(2 * v + 2 * w) + 4 * sin(2 * v))
        )

    def d2w2(self, v: np.ndarray, w: np.ndarray) -> np.ndarray:
        """d^2V/dw^2."""
        c0, c1, c2, c3, c4, c5 = self.c
        return (
            c1 * (-cos(w) - cos(v + w))
            + c2 * (-4 * cos(v + 2 * w) - cos(v - w) - cos(2 * v + w))
            + c3 * (-4 * cos(2 * w) - 4 * cos(2 * v + 2 * w))
            + c4 * (-sin(w) - sin(v + w))
            + c5 * (-4 * sin(2 * v + 2 * w) - 4 * sin(2 * w))
        )

    def d2vw(self, v: np.ndarray, w: np.ndarray) -> np.ndarray:
        """d^2V/dvdw."""
        c0, c1, c2, c3, c4, c5 = self.c
        return (
            c1 * (-cos(v + w))
            + c2 * (-2 * cos(v + 2 * w) + cos(v - w) - 2 * cos(2 * v + w))
            + c3 * (-4 * cos(2 * v + 2 * w))
            + c4 * (sin(v + w))
            + c5 * (4 * sin(2 * v + 2 * w))
        )

    @property
    def minimum_value(self) -> float:
        """Minimum of V over all (v, w). Cached after first call."""
        if self._min_val is None:
            # Sample on a grid to find approximate minimum
            vv = np.linspace(0, 2 * np.pi, 100)
            vg, wg = np.meshgrid(vv, vv)
            vals = self(vg, wg)
            self._min_val = float(np.min(vals))
        return self._min_val

    def saddle_point_energy(self) -> float:
        """Energy at the saddle point (SP stacking).

        For the hexagonal GSFE, the SP lies along v = w. We maximize V(v, v)
        in the range [2*pi/3, 4*pi/3] to find the saddle point between
        adjacent AB/BA domains.
        """
        def neg_V_diagonal(x):
            return -float(self(np.array(x), np.array(x)))

        result = minimize_scalar(
            neg_V_diagonal, bounds=(2 * np.pi / 3, 4 * np.pi / 3), method="bounded"
        )
        return -result.fun - self.minimum_value
