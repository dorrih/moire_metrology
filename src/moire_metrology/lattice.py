"""Hexagonal lattice geometry and moire superlattice calculations.

Conventions:
    - Angles in degrees externally, radians internally
    - Lattice constants in nm
    - b1, b2 are unit vectors at angles theta0 and theta0 + 60 degrees
"""

from __future__ import annotations

import numpy as np


def rotation_matrix(angle_deg: float) -> np.ndarray:
    """2x2 rotation matrix for a given angle in degrees."""
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


class HexagonalLattice:
    """A 2D hexagonal lattice.

    Parameters
    ----------
    alpha : float
        Lattice constant in nm.
    theta0 : float
        Orientation angle of the lattice in degrees (angle of b1 from x-axis).
    """

    def __init__(self, alpha: float, theta0: float = 0.0):
        self.alpha = alpha
        self.theta0 = theta0

    @property
    def b1(self) -> np.ndarray:
        """First lattice unit vector (length 1)."""
        t = np.radians(self.theta0)
        return np.array([np.cos(t), np.sin(t)])

    @property
    def b2(self) -> np.ndarray:
        """Second lattice unit vector (length 1), at 60 degrees from b1."""
        t = np.radians(self.theta0 + 60.0)
        return np.array([np.cos(t), np.sin(t)])

    @property
    def basis_matrix(self) -> np.ndarray:
        """B = alpha * [b1 | b2], shape (2, 2). Columns are lattice vectors."""
        return self.alpha * np.column_stack([self.b1, self.b2])

    @property
    def reciprocal_matrix(self) -> np.ndarray:
        """M = (2*pi/alpha) * inv([b1 | b2]), shape (2, 2).

        Maps real-space coordinates to stacking phase coordinates:
            [v; w] = M @ [x; y]
        """
        B_unit = np.column_stack([self.b1, self.b2])
        return (2 * np.pi / self.alpha) * np.linalg.inv(B_unit)

    @property
    def unit_cell_area(self) -> float:
        """Unit cell area in nm^2."""
        return np.sqrt(3) / 2 * self.alpha**2


class MoireGeometry:
    """Geometry of a moire superlattice from two hexagonal layers.

    Parameters
    ----------
    lattice : HexagonalLattice
        The reference lattice (layer 2 / substrate).
    theta_twist : float
        Twist angle in degrees between the two layers.
    delta : float
        Lattice mismatch: (alpha_layer1 / alpha_layer2) - 1.
        For homostructures (e.g., TBLG), delta = 0.
    """

    def __init__(self, lattice: HexagonalLattice, theta_twist: float, delta: float = 0.0):
        self.lattice = lattice
        self.theta_twist = theta_twist
        self.delta = delta

    @property
    def R_twist(self) -> np.ndarray:
        """Rotation matrix R(-theta_twist)."""
        return rotation_matrix(-self.theta_twist)

    @property
    def moire_matrix(self) -> np.ndarray:
        """Mr = R(-theta) - (1+delta)*I, shape (2,2).

        The moire vectors satisfy: [V1|V2] = inv(Mr) * B.
        """
        return self.R_twist - (1.0 + self.delta) * np.eye(2)

    @property
    def moire_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """Moire superlattice vectors V1, V2 in nm."""
        B = self.lattice.basis_matrix
        Mr = self.moire_matrix
        V = np.linalg.solve(Mr, B)
        return V[:, 0], V[:, 1]

    @property
    def V1(self) -> np.ndarray:
        return self.moire_vectors[0]

    @property
    def V2(self) -> np.ndarray:
        return self.moire_vectors[1]

    @property
    def wavelength(self) -> float:
        """Moire wavelength in nm (geometric mean of |V1| and |V2|)."""
        V1, V2 = self.moire_vectors
        return np.sqrt(np.linalg.norm(V1) * np.linalg.norm(V2))

    @property
    def Mu1(self) -> np.ndarray:
        """Conversion matrix for layer 1 (twisted layer) stacking phases.

        Maps displacement (ux, uy) to stacking phase change via:
            delta_v = -Mu1[0,:] @ [ux, uy]
            delta_w = -Mu1[1,:] @ [ux, uy]
        """
        M = self.lattice.reciprocal_matrix
        return M @ self.R_twist.T

    @property
    def Mu2(self) -> np.ndarray:
        """Conversion matrix for layer 2 (substrate) stacking phases.

        For the substrate layer, displacement shifts stacking phase as:
            delta_v = Mu2[0,:] @ [ux, uy]
            delta_w = Mu2[1,:] @ [ux, uy]
        """
        M = self.lattice.reciprocal_matrix
        return (1.0 + self.delta) * M

    def stacking_phases(
        self, x: np.ndarray, y: np.ndarray, ux: np.ndarray = None, uy: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute stacking phase coordinates (v, w) at positions (x, y).

        Parameters
        ----------
        x, y : ndarray
            Spatial coordinates in nm.
        ux, uy : ndarray or None
            Displacement field components. If None, assumed zero (unrelaxed).

        Returns
        -------
        v, w : ndarray
            Stacking phase coordinates.
        """
        # Stacking phase = M_twisted @ (r - u) - (1+delta)*M @ r
        # where M_twisted = M @ R(-theta)^T (precomputed as self.Mu1/Mu2)
        pos = np.stack([x, y], axis=0)  # (2, N)

        # Phase from twisted layer
        if ux is not None and uy is not None:
            disp = np.stack([ux, uy], axis=0)
            shifted = pos - disp
        else:
            shifted = pos

        phase_twisted = self.Mu1 @ shifted
        phase_substrate = self.Mu2 @ pos

        phase = phase_twisted - phase_substrate
        return phase[0], phase[1]
