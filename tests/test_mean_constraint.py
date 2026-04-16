"""Tests for MeanDisplacementConstraint and RotationConstraint."""

from __future__ import annotations

import numpy as np
import pytest

from moire_metrology import (
    GRAPHENE_GRAPHENE,
    MeanDisplacementConstraint,
    RotationConstraint,
    RelaxationSolver,
    SolverConfig,
    generate_finite_mesh,
)
from moire_metrology.discretization import Discretization, PinnedConstraints
from moire_metrology.lattice import HexagonalLattice, MoireGeometry
from moire_metrology.mean_constraint import stack_mean_constraints


def _build_finite(theta=1.5, n_cells=2, pixel_size=1.2,
                  nlayer1=1, nlayer2=1):
    mat = GRAPHENE_GRAPHENE.bottom
    lat = HexagonalLattice(alpha=mat.lattice_constant)
    geom = MoireGeometry(lat, theta_twist=theta, delta=0.0)
    mesh = generate_finite_mesh(geom, n_cells=n_cells, pixel_size=pixel_size)
    disc = Discretization(mesh, geom)
    conv = disc.build_conversion_matrices(nlayer1=nlayer1, nlayer2=nlayer2)
    return mesh, geom, disc, conv


def test_build_matrix_shape_and_target():
    _, _, _, conv = _build_finite()
    mdc = MeanDisplacementConstraint.from_layer(
        conv, layer_idx=0, target=(0.1, -0.2),
    )
    B, t = mdc.build_matrix(conv)
    assert B.shape == (2, conv.n_sol)
    assert t.tolist() == [0.1, -0.2]
    # Row 0 should sum to 1 (Nv × 1/Nv coefficients on x-DOFs only).
    assert B[0].sum() == pytest.approx(1.0, rel=1e-12)
    # Row 0 must have nonzeros only at layer-0 x-DOFs.
    Nv = conv.n_vertices
    assert set(B[0].indices.tolist()) == set(range(0, Nv))


def test_stack_rejects_pinned_overlap():
    _, _, _, conv = _build_finite()
    # Pin layer-0 x-DOF at vertex 0, then try to constrain layer-0 mean.
    pi = np.array([0], dtype=int)
    fi = np.array(sorted(set(range(conv.n_sol)) - {0}), dtype=int)
    pc = PinnedConstraints(
        fi, pi, np.zeros(1), len(fi), conv.n_sol,
    )
    mdc = MeanDisplacementConstraint.from_layer(conv, layer_idx=0)
    with pytest.raises(ValueError, match="pinned DOF"):
        stack_mean_constraints([mdc], conv, pinned_constraints=pc)


def test_mean_constraint_enforced_at_solution():
    """After relaxation with a mean constraint, the constraint must hold
    to numerical precision, and the result's energy must be finite and
    below the unrelaxed energy (nontrivial relaxation)."""
    mesh, _, _, conv = _build_finite(theta=1.5, n_cells=2, pixel_size=1.2)
    # Pin substrate fully. Then constrain top-flake mean displacement to 0.
    Nv = conv.n_vertices
    pinned = {Nv + v for v in range(Nv)} | {3*Nv + v for v in range(Nv)}
    pi = np.array(sorted(pinned), dtype=int)
    fi = np.array(sorted(set(range(conv.n_sol)) - pinned), dtype=int)
    pc = PinnedConstraints(fi, pi, np.zeros(len(pi)), len(fi), conv.n_sol)
    mdc = MeanDisplacementConstraint.from_layer(
        conv, layer_idx=0, target=(0.0, 0.0),
    )

    cfg = SolverConfig(
        method="newton", display=False,
        elastic_strain="cauchy",
        max_iter=200, gtol=1e-4, rtol=1e-6, etol=1e-9, etol_window=10,
    )
    r = RelaxationSolver(cfg).solve(
        moire_interface=GRAPHENE_GRAPHENE,
        theta_twist=1.5, delta=0.0,
        mesh=mesh, constraints=pc, mean_constraints=[mdc],
    )
    # Energy is finite and relaxation is nontrivial.
    assert np.isfinite(r.total_energy)
    assert r.total_energy < r.unrelaxed_energy

    # Reconstruct full U and check mean.
    U_free = r.optimizer_result.x
    U_full = np.zeros(conv.n_sol)
    U_full[pc.free_indices] = U_free
    ux1 = U_full[0:Nv]
    uy1 = U_full[2*Nv:3*Nv]
    # Mean displacement of top flake must vanish to tight tolerance.
    assert abs(ux1.mean()) < 1e-8, f"mean ux1 = {ux1.mean()}"
    assert abs(uy1.mean()) < 1e-8, f"mean uy1 = {uy1.mean()}"


def test_mean_constraint_differs_from_unconstrained_with_nonzero_target():
    """If we set target != 0, the resulting mean must match the target
    and the relaxed energy should differ from the target=0 case."""
    mesh, _, _, conv = _build_finite(theta=1.5, n_cells=2, pixel_size=1.2)
    Nv = conv.n_vertices
    pinned = {Nv + v for v in range(Nv)} | {3*Nv + v for v in range(Nv)}
    pi = np.array(sorted(pinned), dtype=int)
    fi = np.array(sorted(set(range(conv.n_sol)) - pinned), dtype=int)
    pc = PinnedConstraints(fi, pi, np.zeros(len(pi)), len(fi), conv.n_sol)

    cfg = SolverConfig(method="newton", display=False,
                       elastic_strain="cauchy",
                       max_iter=200, gtol=1e-4, rtol=1e-6,
                       etol=1e-9, etol_window=10)
    solver = RelaxationSolver(cfg)

    # target=0
    mdc0 = MeanDisplacementConstraint.from_layer(conv, 0, target=(0.0, 0.0))
    r0 = solver.solve(moire_interface=GRAPHENE_GRAPHENE,
                      theta_twist=1.5, delta=0.0,
                      mesh=mesh, constraints=pc, mean_constraints=[mdc0])

    # target nonzero
    mdc1 = MeanDisplacementConstraint.from_layer(conv, 0,
                                                  target=(0.02, 0.0))
    r1 = solver.solve(moire_interface=GRAPHENE_GRAPHENE,
                      theta_twist=1.5, delta=0.0,
                      mesh=mesh, constraints=pc, mean_constraints=[mdc1])

    U1_free = r1.optimizer_result.x
    U1 = np.zeros(conv.n_sol)
    U1[pc.free_indices] = U1_free
    ux1 = U1[0:Nv]
    uy1 = U1[2*Nv:3*Nv]
    assert abs(ux1.mean() - 0.02) < 1e-8
    assert abs(uy1.mean()) < 1e-8
    # Energies should be different (constraint is active).
    assert abs(r0.total_energy - r1.total_energy) > 1e-4


# --- RotationConstraint tests ---


def test_rotation_constraint_build_matrix():
    """Matrix has the right shape and target = 0."""
    mesh, _, _, conv = _build_finite()
    rot = RotationConstraint.from_layer(conv, mesh_points=mesh.points, layer_idx=0)
    B, t = rot.build_matrix(conv)
    assert B.shape == (1, conv.n_sol)
    assert t.tolist() == [0.0]
    # Nonzero entries should touch both x- and y-DOFs of layer 0.
    Nv = conv.n_vertices
    nnz_cols = set(B.indices.tolist())
    x_dofs = set(range(0, Nv))
    y_dofs = set(range(2 * Nv, 3 * Nv))
    assert nnz_cols <= (x_dofs | y_dofs)
    assert len(nnz_cols & x_dofs) > 0
    assert len(nnz_cols & y_dofs) > 0


def test_rotation_constraint_suppresses_rotation():
    """With MDC + RotationConstraint, the net rotation of the top flake
    should be much smaller than with MDC alone."""
    mesh, _, _, conv = _build_finite(theta=1.5, n_cells=2, pixel_size=1.2)
    Nv = conv.n_vertices
    pinned = {Nv + v for v in range(Nv)} | {3 * Nv + v for v in range(Nv)}
    pi = np.array(sorted(pinned), dtype=int)
    fi = np.array(sorted(set(range(conv.n_sol)) - pinned), dtype=int)
    pc = PinnedConstraints(fi, pi, np.zeros(len(pi)), len(fi), conv.n_sol)

    mdc = MeanDisplacementConstraint.from_layer(conv, layer_idx=0)
    rot = RotationConstraint.from_layer(conv, mesh_points=mesh.points, layer_idx=0)

    cfg = SolverConfig(method="newton", display=False, elastic_strain="cauchy",
                       max_iter=200, gtol=1e-4, rtol=1e-6,
                       etol=1e-9, etol_window=10)
    solver = RelaxationSolver(cfg)

    r_mdc = solver.solve(moire_interface=GRAPHENE_GRAPHENE,
                         theta_twist=1.5, delta=0.0,
                         mesh=mesh, constraints=pc,
                         mean_constraints=[mdc])

    r_both = solver.solve(moire_interface=GRAPHENE_GRAPHENE,
                          theta_twist=1.5, delta=0.0,
                          mesh=mesh, constraints=pc,
                          mean_constraints=[mdc, rot])

    def _net_rotation(U_free):
        U_full = np.zeros(conv.n_sol)
        U_full[pc.free_indices] = U_free
        ux = U_full[:Nv]
        uy = U_full[2 * Nv:3 * Nv]
        x = mesh.points[0] - mesh.points[0].mean()
        y = mesh.points[1] - mesh.points[1].mean()
        A = np.column_stack([np.ones(Nv), x, y])
        sx = np.linalg.lstsq(A, ux, rcond=None)[0]
        sy = np.linalg.lstsq(A, uy, rcond=None)[0]
        return 0.5 * (sy[1] - sx[2])  # rotation in radians

    rot_mdc = abs(_net_rotation(r_mdc.optimizer_result.x))
    rot_both = abs(_net_rotation(r_both.optimizer_result.x))

    # Rotation with the constraint should be at least 2× smaller.
    assert rot_both < rot_mdc * 0.5, (
        f"RotationConstraint did not suppress rotation: "
        f"|rot_mdc|={rot_mdc:.2e}, |rot_both|={rot_both:.2e}"
    )
    # Both solutions must be nontrivially relaxed.
    assert np.isfinite(r_both.total_energy)
    assert r_both.total_energy < r_both.unrelaxed_energy


def test_rotation_constraint_stacks_with_mdc():
    """RotationConstraint can be stacked with MDC via stack_mean_constraints."""
    mesh, _, _, conv = _build_finite()
    Nv = conv.n_vertices
    pinned = {Nv + v for v in range(Nv)} | {3 * Nv + v for v in range(Nv)}
    pi = np.array(sorted(pinned), dtype=int)
    fi = np.array(sorted(set(range(conv.n_sol)) - pinned), dtype=int)
    pc = PinnedConstraints(fi, pi, np.zeros(len(pi)), len(fi), conv.n_sol)

    mdc = MeanDisplacementConstraint.from_layer(conv, layer_idx=0)
    rot = RotationConstraint.from_layer(conv, mesh_points=mesh.points, layer_idx=0)
    B, t = stack_mean_constraints([mdc, rot], conv, pinned_constraints=pc)
    # 2 rows from MDC + 1 from rotation = 3 total
    assert B.shape[0] == 3
    assert len(t) == 3
    assert t[2] == 0.0  # rotation target
