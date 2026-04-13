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

    # Deformation matrix and strain tensor.
    # NOTE: ``b1b2`` is already premultiplied by alpha at construction
    # (lines above), so the second term is just ``b1b2 @ inv(v1v2)`` —
    # an earlier version of this file applied the alpha factor twice
    # here, which made M off-scale and broke the recovered strain tensor
    # (eps_s came out wrong at the unstrained-symmetric point).
    M = (1.0 + delta) * np.eye(2) + b1b2 @ np.linalg.inv(v1v2)
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
    """Extract strain with phi0 chosen to minimize compression strain \|eps_c\|.

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


def compute_strain_field(
    x: np.ndarray,
    y: np.ndarray,
    I_field,
    J_field,
    alpha1: float,
    alpha2: float,
    phi0_deg: float,
) -> dict[str, np.ndarray]:
    """Compute the spatially-varying strain field from registry polynomials.

    Implements the spatial extension of the ACS Nano paper strain
    inversion. At each query point, the local moiré reciprocal lattice
    vectors are obtained from the polynomial fit gradients via eq. 9:

        [v1(r), v2(r)] = inv([∇I(r); ∇J(r)])

    yielding `(λ1, λ2, φ1, φ2)` at every point. The pointwise
    :func:`get_strain` is then called with a global ``phi0_deg`` to
    recover `(θ, ε_c, ε_s)` per point.

    Parameters
    ----------
    x, y : ndarray
        Query coordinates (nm), of any common shape.
    I_field, J_field : RegistryField
        Polynomial fits of the moire registry index fields, with
        analytic ``dx`` / ``dy`` evaluators (e.g.
        :class:`moire_metrology.strain.RegistryField`).
    alpha1, alpha2 : float
        Lattice constants of the two layers (nm). Convention follows
        :func:`get_strain` — ``alpha2`` is the larger of the two for
        the H-MoSe2/WSe2 sample reproduced in the paper.
    phi0_deg : float
        Substrate lattice orientation angle (degrees), held constant
        across the field. The paper uses a global ``φ0`` rather than
        a per-point one.

    Returns
    -------
    dict
        Each entry has the same shape as ``x``. Keys:

        ``theta``, ``eps_c``, ``eps_s``
            Local twist angle (deg) and the volumetric / shear strain
            scalars used in the paper figures.
        ``S11``, ``S12``, ``S22``
            Components of the symmetric strain tensor in the global
            ``(x, y)`` frame. ``S21 = S12``. These are the natural
            inputs to a gradient-integration IC builder for the
            relaxation framework.
        ``eps1``, ``eps2``, ``strain_angle``
            Principal strains and the principal-axis angle (deg).
        ``lambda1``, ``lambda2``, ``phi1_deg``, ``phi2_deg``
            The local moire vector lengths and orientations recovered
            from the eq. 9 inversion (intermediate quantities, exposed
            for diagnostics).
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    shape = x_arr.shape
    xf = x_arr.ravel()
    yf = y_arr.ravel()

    dIdx = I_field.dx(xf, yf)
    dIdy = I_field.dy(xf, yf)
    dJdx = J_field.dx(xf, yf)
    dJdy = J_field.dy(xf, yf)

    # Eq. 9 of the ACS Nano paper (matches MATLAB
    # strain_extraction_2D_spatial_ver2.m lines 179-183 with no global
    # sign flip):
    #     v1 = ( dJ/dy / D, -dJ/dx / D )
    #     v2 = (-dI/dy / D,  dI/dx / D )
    D = dIdx * dJdy - dIdy * dJdx
    v1x = dJdy / D
    v1y = -dJdx / D
    v2x = -dIdy / D
    v2y = dIdx / D

    lambda1 = np.hypot(v1x, v1y)
    lambda2 = np.hypot(v2x, v2y)
    phi1_deg = np.degrees(np.arctan2(v1y, v1x))
    phi2_deg = np.degrees(np.arctan2(v2y, v2x))

    # Vectorized strain inversion (replaces per-point get_strain loop).
    # All operations below are on arrays of length n = xf.size.
    alpha = alpha2
    delta = alpha2 / alpha1 - 1.0
    phi0_rad = np.radians(phi0_deg)
    phi1_rad = np.radians(phi1_deg)
    phi2_rad = np.radians(phi2_deg)

    # b1b2 is constant (2, 2) — depends only on phi0 and alpha.
    b1b2 = alpha * np.array([
        [cos(phi0_rad), cos(phi0_rad + pi / 3)],
        [sin(phi0_rad), sin(phi0_rad + pi / 3)],
    ])

    # v1v2 per point: columns are the moiré vectors.
    # v1v2[0,0] = lambda1*cos(phi1), v1v2[0,1] = lambda2*cos(phi2), etc.
    v1v2_00 = lambda1 * np.cos(phi1_rad)
    v1v2_01 = lambda2 * np.cos(phi2_rad)
    v1v2_10 = lambda1 * np.sin(phi1_rad)
    v1v2_11 = lambda2 * np.sin(phi2_rad)

    # Batch 2x2 inverse: inv([[a,b],[c,d]]) = (1/det)*[[d,-b],[-c,a]]
    det_v = v1v2_00 * v1v2_11 - v1v2_01 * v1v2_10
    inv_det = 1.0 / det_v
    inv_00 = v1v2_11 * inv_det
    inv_01 = -v1v2_01 * inv_det
    inv_10 = -v1v2_10 * inv_det
    inv_11 = v1v2_00 * inv_det

    # b1b2 @ inv(v1v2) per point — (2,2) @ (2,2) batch
    bi_00 = b1b2[0, 0] * inv_00 + b1b2[0, 1] * inv_10
    bi_01 = b1b2[0, 0] * inv_01 + b1b2[0, 1] * inv_11
    bi_10 = b1b2[1, 0] * inv_00 + b1b2[1, 1] * inv_10
    bi_11 = b1b2[1, 0] * inv_01 + b1b2[1, 1] * inv_11

    # M = (1+delta)*I + b1b2 @ inv(v1v2)
    M_00 = (1.0 + delta) + bi_00
    M_01 = bi_01
    M_10 = bi_10
    M_11 = (1.0 + delta) + bi_11

    # Twist angle from moiré vector geometry (same formula as get_strain)
    dphi = phi2_rad - phi1_rad
    x0 = 2.0 * (1.0 + delta) * lambda1 * lambda2 * np.sin(dphi) / alpha
    r = np.sqrt(
        lambda1**2 + lambda2**2
        - 2.0 * lambda1 * lambda2 * np.cos(dphi + pi / 3)
    )
    dphi0 = np.arctan2(
        lambda1 * np.cos(phi1_rad - pi / 3) - lambda2 * np.cos(phi2_rad),
        -lambda1 * np.sin(phi1_rad - pi / 3) + lambda2 * np.sin(phi2_rad),
    )
    x = x0 + r * np.cos(dphi0 - phi0_rad)
    y = r * np.sin(dphi0 - phi0_rad)
    theta_rad = np.arctan2(y, x)
    theta = np.degrees(theta_rad)

    # Rotation R(theta) per point
    ct = np.cos(theta_rad)
    st = np.sin(theta_rad)

    # S = I - R(theta) @ M  (per point)
    RM_00 = ct * M_00 - st * M_10
    RM_01 = ct * M_01 - st * M_11
    RM_11 = st * M_01 + ct * M_11
    S11 = 1.0 - RM_00
    S12 = -RM_01
    S22 = 1.0 - RM_11

    # Principal strains (vectorized get_strain_axis)
    trace = S11 + S22
    det_S = S11 * S22 - S12**2
    discriminant = np.maximum(trace**2 - 4.0 * det_S, 0.0)
    d = np.sqrt(discriminant)
    eps1 = (trace + d) / 2.0
    eps2 = (trace - d) / 2.0
    eps_c = (eps1 + eps2) / 2.0
    eps_s = (eps1 - eps2) / 2.0
    strain_angle = np.degrees(
        np.arctan2(S12, S11 - 0.5 * (eps1 + eps2))
    )

    return {
        "theta": theta.reshape(shape),
        "eps_c": eps_c.reshape(shape),
        "eps_s": eps_s.reshape(shape),
        "S11": S11.reshape(shape),
        "S12": S12.reshape(shape),
        "S22": S22.reshape(shape),
        "eps1": eps1.reshape(shape),
        "eps2": eps2.reshape(shape),
        "strain_angle": strain_angle.reshape(shape),
        "lambda1": lambda1.reshape(shape),
        "lambda2": lambda2.reshape(shape),
        "phi1_deg": phi1_deg.reshape(shape),
        "phi2_deg": phi2_deg.reshape(shape),
    }


def displacement_from_strain_field(
    discretization,
    *,
    theta_deg: np.ndarray,
    theta_avg_deg: float,
    S11: np.ndarray,
    S12: np.ndarray,
    S22: np.ndarray,
    pin_vertex: int = 0,
    dirichlet_vertices: np.ndarray | None = None,
    dirichlet_ux: np.ndarray | None = None,
    dirichlet_uy: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a displacement field from a target local twist + strain.

    Translates the spatially-varying ``(θ(r), S(r))`` output of
    :func:`compute_strain_field` into the relaxation framework's native
    ``u(r)`` language. The framework holds a single global twist
    ``θ_avg`` fixed, so any local twist deviation ``δθ(r) = θ(r) − θ_avg``
    and any local strain ``S(r)`` must show up as a gradient of ``u``.
    Specifically (with the framework's eps=0.5 layer partition,
    confirmed against ``energy.py`` and ``discretization.py``):

        ∂u_x/∂x = S11(r)
        ∂u_y/∂y = S22(r)
        ∂u_x/∂y = S12(r) − δθ(r)
        ∂u_y/∂x = S12(r) + δθ(r)

    These four equations are over-determined (four constraints on the
    four components of ``∂u/∂r`` per point — exactly the count, but
    with a global compatibility condition that need not hold for noisy
    measured data). The implementation reads them as a sparse linear
    least-squares problem on the FEM mesh: ``[Dx; Dy] @ u_x = (S11; S12 − δθ)``
    and similarly for ``u_y``, where ``Dx`` and ``Dy`` are
    :class:`Discretization`'s vertex-to-triangle gradient operators.

    When ``dirichlet_vertices`` is given, those vertices are held at
    fixed ``(dirichlet_ux, dirichlet_uy)`` values and the Poisson
    solve optimizes the remaining vertices. This is the natural way
    to combine the gradient IC with pinning constraints: the pinned
    vertices get the displacement that the
    :class:`~moire_metrology.pinning.PinningMap` computes for their
    target stacking, and the free vertices are interpolated to match
    the target gradient field. When no Dirichlet vertices are given,
    a single ``pin_vertex`` is pinned to zero to fix the global
    translation gauge.

    Parameters
    ----------
    discretization : Discretization
        FEM discretization carrying the mesh and the gradient operators.
    theta_deg : ndarray, shape (Nv,)
        Local twist angle at each mesh vertex (degrees). Use
        ``np.abs(compute_strain_field(...)["theta"])`` since the
        package convention reports ``-θ``.
    theta_avg_deg : float
        Global twist angle the relaxation framework holds fixed (deg).
    S11, S12, S22 : ndarray, shape (Nv,)
        Strain tensor components at each mesh vertex (the
        ``compute_strain_field`` output keys of the same names).
    pin_vertex : int
        Index of the vertex pinned to ``u = 0`` to fix the global
        translation gauge. Ignored when ``dirichlet_vertices`` is set.
    dirichlet_vertices : ndarray of int or None
        Vertex indices where ``u`` is prescribed. When given,
        ``dirichlet_ux`` and ``dirichlet_uy`` must also be provided.
    dirichlet_ux, dirichlet_uy : ndarray or None
        Prescribed displacement values at the Dirichlet vertices (nm).

    Returns
    -------
    ux, uy : ndarray, shape (Nv,)
        Per-vertex displacement field components (nm), suitable as the
        initial condition for a relaxation against the average-twist
        moiré geometry.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    Dx = discretization.diff_mat_x  # (Nt, Nv)
    Dy = discretization.diff_mat_y
    Nt, Nv = Dx.shape

    triangles = discretization.mesh.triangles  # (Nt, 3)

    def vert_to_tri(v: np.ndarray) -> np.ndarray:
        return v[triangles].mean(axis=1)

    delta_theta_rad = np.radians(vert_to_tri(theta_deg) - theta_avg_deg)
    s11_t = vert_to_tri(S11)
    s12_t = vert_to_tri(S12)
    s22_t = vert_to_tri(S22)

    # Per-triangle target gradients of u_x and u_y.
    g_uxx = s11_t
    g_uxy = s12_t - delta_theta_rad
    g_uyx = s12_t + delta_theta_rad
    g_uyy = s22_t

    # Stack the two gradient equations into a single least-squares system
    # per displacement component.
    A = sp.vstack([Dx, Dy]).tocsr()  # (2Nt, Nv)
    b_x = np.concatenate([g_uxx, g_uxy])
    b_y = np.concatenate([g_uyx, g_uyy])

    # Determine which vertices are fixed (Dirichlet BCs).
    ux = np.zeros(Nv)
    uy = np.zeros(Nv)
    fixed = np.zeros(Nv, dtype=bool)

    if dirichlet_vertices is not None:
        dv = np.asarray(dirichlet_vertices, dtype=int)
        ux[dv] = np.asarray(dirichlet_ux)
        uy[dv] = np.asarray(dirichlet_uy)
        fixed[dv] = True
    else:
        fixed[int(pin_vertex)] = True

    free = ~fixed
    A_free = A[:, free]

    # Move the known (Dirichlet) contributions to the right-hand side.
    A_fixed = A[:, fixed]
    rhs_x = b_x - A_fixed @ ux[fixed]
    rhs_y = b_y - A_fixed @ uy[fixed]

    res_x = spla.lsmr(A_free, rhs_x, atol=1e-12, btol=1e-12)
    res_y = spla.lsmr(A_free, rhs_y, atol=1e-12, btol=1e-12)

    ux[free] = res_x[0]
    uy[free] = res_y[0]
    return ux, uy


def compute_displacement_field(
    x: np.ndarray,
    y: np.ndarray,
    I_field,
    J_field,
    geometry,
    target_stacking: str = "BA",
    remove_mean: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a displacement field from polynomial registry fits.

    The pointwise pinning machinery in :mod:`moire_metrology.pinning`
    solves ``Mu @ u = -(v_target - v0)`` at individual sites to put a
    given mesh vertex at a chosen stacking. This function does the same
    solve at every query point ``(x, y)``, but with ``v_target`` coming
    from a continuous polynomial fit of the registry fields rather than
    discrete pin assignments. The result is a smooth displacement field
    suitable as the initial condition for relaxation against an
    experimentally measured moiré pattern.

    The target phase at each point is

        v_target(r) = 2π · I(r) + v_offset
        w_target(r) = 2π · J(r) + w_offset

    where ``(v_offset, w_offset)`` is the constant phase shift from
    :data:`moire_metrology.pinning.STACKING_PHASES` that puts integer
    ``(I, J)`` data sites at the requested ``target_stacking`` (e.g.
    ``"BA"`` is the global GSFE minimum for H-stacked TMDs).

    Parameters
    ----------
    x, y : ndarray
        Query coordinates (nm), of any common shape.
    I_field, J_field : RegistryField
        Polynomial fits of the moire registry index fields.
    geometry : MoireGeometry
        The average moire geometry. Provides ``stacking_phases`` and
        ``Mu1`` for the per-point linear solve.
    target_stacking : str
        One of ``"AA"``, ``"AB"``, ``"BA"``. Default ``"BA"``.
    remove_mean : bool
        If True, subtract the mean of the resulting displacement so the
        global translation gauge is fixed at zero. Default True.

    Returns
    -------
    ux, uy : ndarray
        Displacement field components (nm), of the same shape as ``x``.

    Notes
    -----
    The polynomial fits are unconstrained outside the convex hull of
    their training data and grow rapidly there. Mask the query points
    with :func:`convex_hull_mask` before calling this function on a
    mesh that extends past the data extent.
    """
    from ..pinning import STACKING_PHASES

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    shape = x_arr.shape
    xf = x_arr.ravel()
    yf = y_arr.ravel()

    if target_stacking not in STACKING_PHASES:
        raise ValueError(
            f"Unknown stacking '{target_stacking}'. "
            f"Use one of: {list(STACKING_PHASES)}"
        )
    v_off, w_off = STACKING_PHASES[target_stacking]

    v0, w0 = geometry.stacking_phases(xf, yf)
    v_target = 2.0 * pi * I_field(xf, yf) + v_off
    w_target = 2.0 * pi * J_field(xf, yf) + w_off

    Mu = geometry.Mu1
    rhs = np.stack([v0 - v_target, w0 - w_target], axis=0)  # (2, N)
    u = np.linalg.solve(Mu, rhs)  # (2, N)
    ux = u[0]
    uy = u[1]

    if remove_mean and ux.size > 0:
        ux = ux - ux.mean()
        uy = uy - uy.mean()

    return ux.reshape(shape), uy.reshape(shape)
