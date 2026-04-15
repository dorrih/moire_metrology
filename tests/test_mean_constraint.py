"""Tests for MeanDisplacementConstraint (Lagrange-multiplier enforcement)."""

from __future__ import annotations

import numpy as np
import pytest

from moire_metrology import (
    GRAPHENE_GRAPHENE,
    MeanDisplacementConstraint,
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
