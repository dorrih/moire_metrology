"""Core strain extraction from moire superlattice observables.

Implements the methodology from:
    Halbertal et al., "Extracting the Strain Matrix and Twist Angle from the
    Moire Superlattice in van der Waals Hetero-Structures"

Given the moire vectors (lambda1, lambda2, phi1, phi2) and lattice parameters,
extracts the twist angle, strain tensor, and principal strains.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy import cos, sin, sqrt, arctan2, arccos, pi


@dataclass
class StrainResult:
    """Result of strain extraction from moire observables.

    Attributes
    ----------
    theta_twist : float
        Twist angle in degrees.
    phi0 : float
        Substrate lattice orientation in degrees.
    S11, S12, S22 : float
        Strain tensor components (symmetric: S21 = S12).
    eps_c : float
        Volumetric (compression) strain = (eps1 + eps2) / 2.
    eps_s : float
        Shear (deviatoric) strain = (eps1 - eps2) / 2.
    eps1, eps2 : float
        Principal strains (eigenvalues of strain tensor).
    strain_angle : float
        Principal strain axis angle in degrees.
    """

    theta_twist: float
    phi0: float
    S11: float
    S12: float
    S22: float
    eps_c: float
    eps_s: float
    eps1: float
    eps2: float
    strain_angle: float


def get_strain(
    alpha1: float,
    alpha2: float,
    lambda1: float,
    lambda2: float,
    phi1_deg: float,
    phi2_deg: float,
    phi0: float,
) -> StrainResult:
    """Extract twist angle and strain from moire observables at a given phi0.

    Parameters
    ----------
    alpha1, alpha2 : float
        Lattice constants of the two layers (nm).
    lambda1, lambda2 : float
        Moire periods along the two moire vectors (nm).
    phi1_deg, phi2_deg : float
        Orientations of the two moire vectors (degrees).
    phi0 : float
        Substrate lattice orientation angle (degrees).

    Returns
    -------
    StrainResult
    """
    alpha = alpha2
    delta = alpha2 / alpha1 - 1.0

    phi1 = np.radians(phi1_deg)
    phi2 = np.radians(phi2_deg)
    phi0_rad = np.radians(phi0)

    # Basis vectors [b1 | b2] (columns)
    b1b2 = alpha * np.array([
        [cos(phi0_rad), cos(phi0_rad + pi / 3)],
        [sin(phi0_rad), sin(phi0_rad + pi / 3)],
    ])

    # Moire vectors [v1 | v2]
    v1v2 = np.array([
        [lambda1 * cos(phi1), lambda2 * cos(phi2)],
        [lambda1 * sin(phi1), lambda2 * sin(phi2)],
    ])

    # Intermediate quantities
    dphi = phi2 - phi1
    x0 = 2.0 * (1.0 + delta) * lambda1 * lambda2 * sin(dphi) / alpha

    r = sqrt(
        lambda1**2 + lambda2**2 - 2.0 * lambda1 * lambda2 * cos(dphi + pi / 3)
    )

    dphi0 = arctan2(
        lambda1 * cos(phi1 - pi / 3) - lambda2 * cos(phi2),
        -lambda1 * sin(phi1 - pi / 3) + lambda2 * sin(phi2),
    )

    x = x0 + r * cos(dphi0 - phi0_rad)
    y = r * sin(dphi0 - phi0_rad)

    # Twist angle
    theta_twist_rad = arctan2(y, x)
    theta_twist_deg = np.degrees(theta_twist_rad)

    # Rotation matrix R(theta)
    ct, st = cos(theta_twist_rad), sin(theta_twist_rad)
    Rt = np.array([[ct, -st], [st, ct]])

    # Deformation matrix and strain tensor
    M = (1.0 + delta) * np.eye(2) + alpha * b1b2 @ np.linalg.inv(v1v2)
    S_mat = np.eye(2) - Rt @ M

    S11 = S_mat[0, 0]
    S12 = S_mat[0, 1]
    S22 = S_mat[1, 1]

    # Principal strains
    eps1, eps2, strain_angle = get_strain_axis(S11, S12, S22)
    eps_c = (eps1 + eps2) / 2.0
    eps_s = (eps1 - eps2) / 2.0

    return StrainResult(
        theta_twist=theta_twist_deg,
        phi0=phi0,
        S11=S11, S12=S12, S22=S22,
        eps_c=eps_c, eps_s=eps_s,
        eps1=eps1, eps2=eps2,
        strain_angle=strain_angle,
    )


def get_strain_minimize_compression(
    alpha1: float,
    alpha2: float,
    lambda1: float,
    lambda2: float,
    phi1_deg: float,
    phi2_deg: float,
    phi0_guess: float = 0.0,
) -> StrainResult:
    """Extract strain with phi0 chosen to minimize compression strain |eps_c|.

    This solves analytically for phi0 such that eps_c = 0 when possible,
    or finds the closest achievable phi0 when eps_c = 0 is infeasible.

    Parameters
    ----------
    alpha1, alpha2 : float
        Lattice constants (nm).
    lambda1, lambda2 : float
        Moire periods (nm).
    phi1_deg, phi2_deg : float
        Moire vector orientations (degrees).
    phi0_guess : float
        Initial guess for phi0 (degrees), used for disambiguation.

    Returns
    -------
    StrainResult
    """
    alpha = alpha2
    delta = alpha2 / alpha1 - 1.0

    phi1 = np.radians(phi1_deg)
    phi2 = np.radians(phi2_deg)
    phi0_guess_rad = np.radians(phi0_guess)

    dphi = phi2 - phi1
    x0 = 2.0 * (1.0 + delta) * lambda1 * lambda2 * sin(dphi) / alpha

    r = sqrt(
        lambda1**2 + lambda2**2 - 2.0 * lambda1 * lambda2 * cos(dphi + pi / 3)
    )

    dphi0 = arctan2(
        lambda1 * cos(phi1 - pi / 3) - lambda2 * cos(phi2),
        -lambda1 * sin(phi1 - pi / 3) + lambda2 * sin(phi2),
    )

    # For eps_c = 0: need R = x0 / (1+delta)
    # R^2 = x0^2 + r^2 + 2*x0*r*cos(dphi0 - phi0)
    # => cos(dphi0 - phi0) = (R_target^2 - x0^2 - r^2) / (2*x0*r)
    R_target = x0 / (1.0 + delta)
    cos_val = (R_target**2 - x0**2 - r**2) / (2.0 * x0 * r) if abs(x0 * r) > 1e-30 else 2.0

    if abs(cos_val) <= 1.0:
        # Feasible: eps_c = 0 achievable
        angle = arccos(cos_val)
        # Two solutions: dphi0 - phi0 = +/- angle
        phi0_candidates = [dphi0 - angle, dphi0 + angle]
        # Pick the one closest to phi0_guess
        dists = [abs(np.angle(np.exp(1j * (c - phi0_guess_rad)))) for c in phi0_candidates]
        phi0_opt = phi0_candidates[np.argmin(dists)]
    else:
        # Infeasible: find phi0 that minimizes |eps_c|
        phi0_candidates = [dphi0 - pi, dphi0, dphi0 + pi]
        best_eps_c = np.inf
        phi0_opt = phi0_candidates[0]

        for phi0_cand in phi0_candidates:
            x = x0 + r * cos(dphi0 - phi0_cand)
            y = r * sin(dphi0 - phi0_cand)
            R = sqrt(x**2 + y**2)
            eps_c = 1.0 - (1.0 + delta) * R / x0 if abs(x0) > 1e-30 else 0.0

            if abs(eps_c) < best_eps_c:
                best_eps_c = abs(eps_c)
                phi0_opt = phi0_cand

    return get_strain(alpha1, alpha2, lambda1, lambda2, phi1_deg, phi2_deg,
                      np.degrees(phi0_opt))


def get_strain_axis(
    S11: float, S12: float, S22: float
) -> tuple[float, float, float]:
    """Diagonalize the strain tensor to get principal strains.

    Parameters
    ----------
    S11, S12, S22 : float
        Strain tensor components (S is symmetric: S21 = S12).

    Returns
    -------
    eps1 : float
        Principal strain 1 (larger).
    eps2 : float
        Principal strain 2 (smaller).
    strain_angle : float
        Principal strain axis angle in degrees.
    """
    trace = S11 + S22
    det = S11 * S22 - S12**2
    discriminant = max(trace**2 - 4.0 * det, 0.0)
    d = sqrt(discriminant)

    eps1 = (trace + d) / 2.0
    eps2 = (trace - d) / 2.0

    strain_angle = np.degrees(arctan2(S12, S11 - 0.5 * (eps1 + eps2)))

    return eps1, eps2, strain_angle


def shear_strain_invariant(
    alpha1: float,
    alpha2: float,
    lambda1: float,
    lambda2: float,
    phi1_deg: float,
    phi2_deg: float,
) -> float:
    """Compute the phi0-independent shear strain from moire observables.

    This is the closed-form formula from the paper:
        eps_s = alpha * r_minus / (2 * lambda1 * lambda2 * sin(dphi))

    where r_minus = sqrt(l1^2 + l2^2 - 2*l1*l2*cos(dphi - pi/3))

    This quantity depends only on the moire observables, not on phi0.

    Parameters
    ----------
    alpha1, alpha2 : float
        Lattice constants (nm).
    lambda1, lambda2 : float
        Moire periods (nm).
    phi1_deg, phi2_deg : float
        Moire vector orientations (degrees).

    Returns
    -------
    float
        Shear strain magnitude (always non-negative).
    """
    alpha = alpha2
    dphi = np.radians(phi2_deg - phi1_deg)

    r_minus = sqrt(
        lambda1**2 + lambda2**2 - 2.0 * lambda1 * lambda2 * cos(dphi - pi / 3)
    )

    denom = 2.0 * lambda1 * lambda2 * abs(sin(dphi))
    if denom < 1e-30:
        return 0.0

    return float(alpha * r_minus / denom)


def compute_displacement_field(
    x: np.ndarray,
    y: np.ndarray,
    I_field: np.ndarray,
    J_field: np.ndarray,
    theta_deg: float,
    theta0_deg: float,
    alpha: float,
    delta: float = 0.0,
    dr: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute displacement field u(x,y) from registry fields I(x,y), J(x,y).

    Parameters
    ----------
    x, y : ndarray
        Spatial coordinates (nm).
    I_field, J_field : ndarray
        Registry field values at each (x, y) point.
    theta_deg : float
        Twist angle (degrees).
    theta0_deg : float
        Lattice orientation angle (degrees).
    alpha : float
        Lattice constant (nm).
    delta : float
        Lattice mismatch.
    dr : ndarray, shape (2,), or None
        Translation offset [dx, dy] in nm.

    Returns
    -------
    ux, uy : ndarray
        Displacement field components (nm).
    """
    theta = np.radians(theta_deg)
    theta0 = np.radians(theta0_deg)

    if dr is None:
        dr = np.zeros(2)

    ct, st = cos(theta), sin(theta)
    Rminus = np.array([[cos(-theta), -sin(-theta)], [sin(-theta), cos(-theta)]])

    b1b2 = alpha * np.array([
        [cos(theta0), cos(theta0 + pi / 3)],
        [sin(theta0), sin(theta0 + pi / 3)],
    ])

    # Prefactor matrices
    A = np.linalg.inv(Rminus + (1.0 + delta) * np.eye(2))
    B_minus = Rminus - (1.0 + delta) * np.eye(2)

    dr_tilde = (np.eye(2) - (1.0 + delta) * np.array([[ct, -st], [st, ct]])) @ dr

    # Compute displacement at each point
    r = np.stack([x, y], axis=0)  # (2, N)
    IJ = np.stack([I_field, J_field], axis=0)  # (2, N)

    u = A @ (2.0 * B_minus @ r - 2.0 * b1b2 @ IJ - 2.0 * Rminus @ dr_tilde[:, None])

    return u[0], u[1]


def compute_strain_field(
    dIdx: np.ndarray,
    dIdy: np.ndarray,
    dJdx: np.ndarray,
    dJdy: np.ndarray,
    theta_deg: float,
    theta0_deg: float,
    alpha: float,
    delta: float = 0.0,
) -> dict[str, np.ndarray]:
    """Compute strain tensor from registry field gradients.

    Parameters
    ----------
    dIdx, dIdy, dJdx, dJdy : ndarray
        Spatial derivatives of the registry fields I(x,y) and J(x,y).
    theta_deg, theta0_deg : float
        Twist and lattice orientation angles (degrees).
    alpha : float
        Lattice constant (nm).
    delta : float
        Lattice mismatch.

    Returns
    -------
    dict with keys:
        'eps_xx', 'eps_xy', 'eps_yy': Strain tensor components.
        'eps1', 'eps2': Principal strains at each point.
        'strain_angle': Principal strain axis angle (degrees).
    """
    theta = np.radians(theta_deg)
    theta0 = np.radians(theta0_deg)

    Rminus = np.array([[cos(-theta), -sin(-theta)], [sin(-theta), cos(-theta)]])
    b1b2 = alpha * np.array([
        [cos(theta0), cos(theta0 + pi / 3)],
        [sin(theta0), sin(theta0 + pi / 3)],
    ])

    A = np.linalg.inv(Rminus + (1.0 + delta) * np.eye(2))
    B_minus = Rminus - (1.0 + delta) * np.eye(2)

    # du/dx = A @ (2*B_minus @ [1;0] - 2*b1b2 @ [dI/dx; dJ/dx])
    e_x = np.array([[1.0], [0.0]])
    e_y = np.array([[0.0], [1.0]])

    term_x_const = 2.0 * B_minus @ e_x  # (2, 1)
    term_y_const = 2.0 * B_minus @ e_y

    # Per-point derivatives
    dIJdx = np.stack([dIdx, dJdx], axis=0)  # (2, N)
    dIJdy = np.stack([dIdy, dJdy], axis=0)

    dudx = A @ (term_x_const - 2.0 * b1b2 @ dIJdx)  # (2, N)
    dudy = A @ (term_y_const - 2.0 * b1b2 @ dIJdy)

    eps_xx = dudx[0]  # dux/dx
    eps_yy = dudy[1]  # duy/dy
    eps_xy = 0.5 * (dudx[1] + dudy[0])  # 0.5*(duy/dx + dux/dy)

    # Principal strains
    trace = eps_xx + eps_yy
    det = eps_xx * eps_yy - eps_xy**2
    discriminant = np.maximum(trace**2 - 4.0 * det, 0.0)
    d = np.sqrt(discriminant)

    eps1 = (trace + d) / 2.0
    eps2 = (trace - d) / 2.0
    strain_angle = np.degrees(np.arctan2(eps_xy, eps_xx - 0.5 * (eps1 + eps2)))

    # Wrap angle to [0, 180)
    strain_angle = strain_angle % 180.0

    return {
        "eps_xx": eps_xx,
        "eps_xy": eps_xy,
        "eps_yy": eps_yy,
        "eps1": eps1,
        "eps2": eps2,
        "strain_angle": strain_angle,
    }
