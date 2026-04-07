"""2D polynomial fitting for moire registry fields.

Fits continuous fields I(x,y) and J(x,y) to traced moire fringe data,
and provides analytical derivatives for strain extraction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RegistryField:
    """A 2D polynomial fit of a moire registry field (I or J).

    The polynomial has the form:
        f(x,y) = sum_{i+j <= order} c_{ij} * x^i * y^j

    Attributes
    ----------
    coeffs : ndarray
        Polynomial coefficients in the order used by _poly_terms.
    order : int
        Maximum polynomial order.
    x_center, y_center : float
        Coordinate centering for numerical stability.
    x_scale, y_scale : float
        Coordinate scaling for numerical stability.
    """

    coeffs: np.ndarray
    order: int
    x_center: float
    y_center: float
    x_scale: float
    y_scale: float

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate the polynomial at (x, y)."""
        xn, yn = self._normalize(x, y)
        A = _poly_terms(xn, yn, self.order)
        return A @ self.coeffs

    def dx(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate df/dx at (x, y)."""
        xn, yn = self._normalize(x, y)
        A = _poly_terms_dx(xn, yn, self.order)
        return (A @ self.coeffs) / self.x_scale

    def dy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate df/dy at (x, y)."""
        xn, yn = self._normalize(x, y)
        A = _poly_terms_dy(xn, yn, self.order)
        return (A @ self.coeffs) / self.y_scale

    def _normalize(self, x, y):
        return (x - self.x_center) / self.x_scale, (y - self.y_center) / self.y_scale

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        values: np.ndarray,
        order: int = 8,
    ) -> RegistryField:
        """Fit a 2D polynomial to scattered data.

        Parameters
        ----------
        x, y : ndarray, shape (N,)
            Data point coordinates (nm).
        values : ndarray, shape (N,)
            Field values at each point (integer registry indices).
        order : int
            Maximum polynomial order (default 8).

        Returns
        -------
        RegistryField
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        values = np.asarray(values, dtype=float)

        # Center and scale for numerical stability
        x_center = np.mean(x)
        y_center = np.mean(y)
        x_scale = max(np.std(x), 1e-10)
        y_scale = max(np.std(y), 1e-10)

        xn = (x - x_center) / x_scale
        yn = (y - y_center) / y_scale

        # Build Vandermonde-like matrix
        A = _poly_terms(xn, yn, order)

        # Least-squares fit
        coeffs, _, _, _ = np.linalg.lstsq(A, values, rcond=None)

        return cls(
            coeffs=coeffs,
            order=order,
            x_center=x_center,
            y_center=y_center,
            x_scale=x_scale,
            y_scale=y_scale,
        )


def _poly_terms(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    """Build the 2D polynomial basis matrix.

    Terms: 1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3, ...
    for all i+j <= order.

    Returns shape (N, n_terms).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    terms = []
    for total in range(order + 1):
        for j in range(total + 1):
            i = total - j
            terms.append(x**i * y**j)
    return np.column_stack(terms)


def _poly_terms_dx(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    """Build the x-derivative of the polynomial basis matrix.

    d/dx(x^i * y^j) = i * x^(i-1) * y^j
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    terms = []
    for total in range(order + 1):
        for j in range(total + 1):
            i = total - j
            if i > 0:
                terms.append(i * x ** (i - 1) * y**j)
            else:
                terms.append(np.zeros_like(x))
    return np.column_stack(terms)


def _poly_terms_dy(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    """Build the y-derivative of the polynomial basis matrix.

    d/dy(x^i * y^j) = j * x^i * y^(j-1)
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    terms = []
    for total in range(order + 1):
        for j in range(total + 1):
            i = total - j
            if j > 0:
                terms.append(j * x**i * y ** (j - 1))
            else:
                terms.append(np.zeros_like(x))
    return np.column_stack(terms)
