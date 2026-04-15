"""Tests for the geometrically nonlinear (St.-Venant–Kirchhoff) elastic term.

Covers:
- Rigid rotation returns zero energy in GL and nonzero in Cauchy
  (regression-protects the rotation-invariance claim).
- Small-strain consistency: GL → Cauchy as |∇u| → 0.
- Gradient vs finite-difference.
- hessp vs finite-difference on the gradient.
- Sparse Hessian agreement with hessp.
- Regression: an end-to-end relaxation with elastic_strain="green_lagrange"
  converges and gives sensibly small differences from Cauchy at small twist.
"""

from __future__ import annotations

import numpy as np
import pytest

from moire_metrology import (
    GRAPHENE_GRAPHENE,
    RelaxationSolver,
    SolverConfig,
    generate_finite_mesh,
)
from moire_metrology.discretization import Discretization
from moire_metrology.energy import RelaxationEnergy
from moire_metrology.gsfe import GSFESurface
from moire_metrology.lattice import HexagonalLattice, MoireGeometry


def _build(strain: str, *, nlayer1: int = 1, nlayer2: int = 1):
    mat_t = GRAPHENE_GRAPHENE.top
    mat_b = GRAPHENE_GRAPHENE.bottom
    lat = HexagonalLattice(alpha=mat_b.lattice_constant)
    geom = MoireGeometry(lat, theta_twist=1.5, delta=0.0)
    mesh = generate_finite_mesh(geom, n_cells=2, pixel_size=1.2)
    disc = Discretization(mesh, geom)
    conv = disc.build_conversion_matrices(nlayer1=nlayer1, nlayer2=nlayer2)
    return mesh, disc, conv, geom, RelaxationEnergy(
        disc=disc, conv=conv, geometry=geom,
        gsfe_interface=GSFESurface(GRAPHENE_GRAPHENE.gsfe_coeffs),
        K1=mat_t.bulk_modulus, G1=mat_t.shear_modulus,
        K2=mat_b.bulk_modulus, G2=mat_b.shear_modulus,
        nlayer1=nlayer1, nlayer2=nlayer2,
        elastic_strain=strain,
    )


def _rigid_rotation_U(mesh, alpha_deg: float, layer_idx: int = 0,
                      n_layers_total: int = 2):
    """Build a full U vector representing a rigid rotation of one layer.

    u(r) = [R(α) − I] · (r − r₀) applied to ``layer_idx``; every other
    layer is zero. Rotation is about the mesh centroid.
    """
    Nv = mesh.n_vertices
    x, y = mesh.points[0], mesh.points[1]
    r0 = np.array([x.mean(), y.mean()])
    a = np.deg2rad(alpha_deg)
    c, s = np.cos(a), np.sin(a)
    ux = (c - 1.0) * (x - r0[0]) - s * (y - r0[1])
    uy = s * (x - r0[0]) + (c - 1.0) * (y - r0[1])

    n_sol = 2 * n_layers_total * Nv
    U = np.zeros(n_sol)
    ox = layer_idx * Nv
    oy = n_layers_total * Nv + layer_idx * Nv
    U[ox:ox + Nv] = ux
    U[oy:oy + Nv] = uy
    return U


def _elastic_only_energy(energy_func: RelaxationEnergy, U: np.ndarray) -> float:
    """Run the functional and subtract the GSFE portion so we read the
    elastic contribution directly. GSFE depends only on the relative
    displacement between the two interface layers; by only rotating one
    layer we keep the GSFE contribution bounded but nonzero, so subtract
    it out by evaluating at U (which has GSFE) vs. U=0 (GSFE of unrelaxed).
    """
    # Simpler and exact: compute elastic directly via the internal path.
    U_full = energy_func._to_full(U)
    if energy_func.elastic_strain == "cauchy":
        return 0.5 * float(U_full.dot(energy_func._H_elastic @ U_full))
    grad = np.zeros_like(U_full)
    return energy_func._gl.energy_grad(U_full, grad)


def test_rigid_rotation_gl_is_zero_cauchy_is_not():
    mesh, *_, ef_c = _build("cauchy")
    mesh2, *_, ef_gl = _build("green_lagrange")
    for alpha in (0.5, 2.0, 10.0):
        U = _rigid_rotation_U(mesh, alpha)
        E_c = _elastic_only_energy(ef_c, U)
        E_gl = _elastic_only_energy(ef_gl, U)
        # Green-Lagrange: rotation-invariant, should be ~machine zero.
        assert abs(E_gl) < 1e-6, f"GL energy at α={alpha}° = {E_gl}"
        # Cauchy: spurious compression, nonzero and grows ~α^4.
        assert E_c > 0, f"Cauchy energy at α={alpha}° should be positive"


def test_small_strain_gl_matches_cauchy():
    """For tiny displacements, GL and Cauchy elastic energies agree."""
    rng = np.random.default_rng(0)
    mesh, *_, ef_c = _build("cauchy")
    mesh2, *_, ef_gl = _build("green_lagrange")
    n = ef_c.conv.n_sol
    U = 1e-4 * rng.standard_normal(n)
    E_c = _elastic_only_energy(ef_c, U)
    E_gl = _elastic_only_energy(ef_gl, U)
    # Relative difference should be ≤ O(|∇u|) ~ 1e-4, with a safety factor.
    assert abs(E_gl - E_c) / max(abs(E_c), 1e-12) < 1e-3


def test_gl_gradient_vs_finite_difference():
    rng = np.random.default_rng(1)
    mesh, *_, ef_gl = _build("green_lagrange")
    n = ef_gl.conv.n_sol
    U = 1e-3 * rng.standard_normal(n)
    _, g = ef_gl(U)
    g_fd = np.zeros_like(U)
    eps = 1e-6
    for i in range(n):
        U[i] += eps
        Ep, _ = ef_gl(U)
        U[i] -= 2 * eps
        Em, _ = ef_gl(U)
        U[i] += eps
        g_fd[i] = (Ep - Em) / (2 * eps)
    np.testing.assert_allclose(g, g_fd, atol=1e-3, rtol=1e-3)


def test_gl_hessp_vs_finite_diff_of_grad():
    rng = np.random.default_rng(2)
    mesh, *_, ef_gl = _build("green_lagrange")
    n = ef_gl.conv.n_sol
    U = 1e-3 * rng.standard_normal(n)
    p = rng.standard_normal(n)
    eps = 1e-6
    _, gp = ef_gl(U + eps * p)
    _, gm = ef_gl(U - eps * p)
    hp_fd = (gp - gm) / (2 * eps)
    hp = ef_gl.hessp(U, p)
    np.testing.assert_allclose(hp, hp_fd, atol=1e-2, rtol=1e-2)


def test_gl_hessian_matches_hessp():
    """Sparse Hessian row i · p == hessp(U, p)[i] for random p."""
    rng = np.random.default_rng(3)
    mesh, *_, ef_gl = _build("green_lagrange")
    n = ef_gl.conv.n_sol
    U = 1e-3 * rng.standard_normal(n)
    H = ef_gl.hessian(U).toarray()
    for i in [0, 5, n // 3, n - 1]:
        e = np.zeros(n); e[i] = 1.0
        np.testing.assert_allclose(H @ e, ef_gl.hessp(U, e), atol=1e-8)


def test_gl_solver_end_to_end_small_twist():
    """End-to-end relaxation with GL strain should converge and land
    close to the Cauchy solution at small twist."""
    cfg_c = SolverConfig(display=False, method="newton", pixel_size=0.5,
                         n_scale=1, elastic_strain="cauchy")
    cfg_gl = SolverConfig(display=False, method="newton", pixel_size=0.5,
                          n_scale=1, elastic_strain="green_lagrange")
    r_c = RelaxationSolver(cfg_c).solve(
        moire_interface=GRAPHENE_GRAPHENE, theta_twist=2.0,
    )
    r_gl = RelaxationSolver(cfg_gl).solve(
        moire_interface=GRAPHENE_GRAPHENE, theta_twist=2.0,
    )
    # Both converged to a finite total energy strictly below the
    # unrelaxed energy (non-trivial relaxation).
    assert np.isfinite(r_c.total_energy)
    assert np.isfinite(r_gl.total_energy)
    assert r_c.total_energy < r_c.unrelaxed_energy
    assert r_gl.total_energy < r_gl.unrelaxed_energy
    # Energies agree to within a few percent at θ = 2°.
    rel = (abs(r_gl.total_energy - r_c.total_energy)
           / abs(r_c.total_energy))
    assert rel < 0.05, f"GL vs Cauchy energy disagree by {rel:.2%}"
