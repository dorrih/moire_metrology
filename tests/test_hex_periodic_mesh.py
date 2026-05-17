"""Tests for the hexagonal Wigner-Seitz supercell mesh + boundary helpers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from moire_metrology import (
    HexagonalLattice,
    MoireGeometry,
    PeriodicPairConstraint,
    generate_hex_periodic_mesh,
    identify_hex_periodic_boundary,
)


def _build_hex_mesh(theta: float = 0.5, pixel_size: float = 2.0):
    lat = HexagonalLattice(alpha=0.247, theta0=0.0)
    geom = MoireGeometry(lat, theta_twist=theta, delta=0.0)
    mesh = generate_hex_periodic_mesh(geom, pixel_size=pixel_size)
    return geom, mesh


def test_hex_mesh_has_six_fold_symmetry():
    """Mesh vertices must be invariant (as a SET) under 60° rotation
    around the cell center.  The cell is centered at AA so this checks
    the C6 symmetry of the moire physics, which the supercell shape
    must respect."""
    _, mesh = _build_hex_mesh()
    pts = mesh.points.T  # (Nv, 2)
    angle = np.pi / 3.0
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    rotated = pts @ R.T
    # Every rotated vertex must coincide with some mesh vertex
    tree = cKDTree(pts)
    d, _ = tree.query(rotated, k=1)
    assert d.max() < 1e-10, (
        f"C6 symmetry violated: max-mismatch under 60° rotation = {d.max():.3e}"
    )


def test_hex_mesh_has_six_corners():
    geom, mesh = _build_hex_mesh()
    info = identify_hex_periodic_boundary(mesh)
    assert len(info["corners"]) == 6
    # The 6 corner vertices must be distinct indices
    assert len(set(info["corners"].tolist())) == 6


def test_hex_mesh_edges_correct_count():
    """The mesh from generate_hex_periodic_mesh, with the triangular
    sub-lattice constructed from V1/N, V2/N where N = round(|V1|/pixel),
    has exactly N+1 vertices per edge (including the two endpoint corners)."""
    geom, mesh = _build_hex_mesh()
    info = identify_hex_periodic_boundary(mesh)
    # All 6 edges should have the same number of vertices
    counts = [len(info["edges"][k]) for k in range(1, 7)]
    assert len(set(counts)) == 1, f"edge vertex counts not uniform: {counts}"
    assert counts[0] >= 3, f"each edge should have at least 3 vertices, got {counts[0]}"


def test_hex_mesh_pairs_match_geometrically():
    """Periodic pairing: for each pair entry, points at src_indices +
    translation must equal points at dst_indices to machine precision."""
    geom, mesh = _build_hex_mesh()
    info = identify_hex_periodic_boundary(mesh)
    for p in info["pairs"]:
        src_pts = mesh.points[:, p["src_indices"]].T
        dst_pts = mesh.points[:, p["dst_indices"]].T
        expected = src_pts + p["translation"]
        err = np.linalg.norm(dst_pts - expected, axis=1).max()
        assert err < 1e-10, (
            f"pair {p['source']}↔{p['dest']}: max position mismatch {err:.3e}"
        )


def test_hex_mesh_three_pairs():
    geom, mesh = _build_hex_mesh()
    info = identify_hex_periodic_boundary(mesh)
    assert len(info["pairs"]) == 3
    # Each pair has the same number of vertices (the pair count = edge count)
    pair_counts = [len(p["src_indices"]) for p in info["pairs"]]
    assert len(set(pair_counts)) == 1, (
        f"pair counts not uniform: {pair_counts}"
    )


def test_hex_periodic_pair_constraint_buildable():
    """The boundary-pair output should plug into PeriodicPairConstraint
    without modification."""
    from moire_metrology.discretization import Discretization
    geom, mesh = _build_hex_mesh()
    info = identify_hex_periodic_boundary(mesh)

    # Build a Discretization + ConversionMatrices so we can call build_matrix
    disc = Discretization(mesh, geom)
    conv = disc.build_conversion_matrices(nlayer1=1, nlayer2=1)

    for p in info["pairs"]:
        pairs = np.column_stack([p["src_indices"], p["dst_indices"]])
        ppc = PeriodicPairConstraint(layer_idx=0, pairs=pairs)
        B, t = ppc.build_matrix(conv)
        # n_rows = N_pairs × 2 components (x and y)
        assert B.shape[0] == 2 * len(p["src_indices"])
        assert B.shape[1] == conv.n_sol
        # Each row should have exactly two nonzeros, +1 and -1
        for i in range(B.shape[0]):
            row = B.getrow(i).toarray().ravel()
            nz = row[row != 0]
            assert nz.size == 2
            assert sorted(nz.tolist()) == [-1.0, 1.0]
        # Target is zero
        assert np.all(t == 0.0)
