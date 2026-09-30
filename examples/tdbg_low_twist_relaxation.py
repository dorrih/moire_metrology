"""TDBG relaxation at low twist (θ = 0.02°).

Twisted double bilayer graphene (TDBG) at very small twist angles develops
a "double domain wall" (2DW) network: circular AB / BA domains bounded by
curved, paired domain walls that converge at AA stacking points. The
pattern reflects the non-zero ``c4, c5`` (AB ↔ BA asymmetric) terms of the
TDBG GSFE, which split AB and BA energies and curve the otherwise-straight
domain walls of the symmetric (TBG) case.

This example reproduces the 2DW relaxation on a hexagonal Wigner-Seitz
periodic supercell of the TDBG-DFTD2 moiré at θ = 0.02° with the bottom
flake pinned at ``U = 0`` (so the top flake carries the entire relaxation
field — the same convention as the published TDBG MATLAB reference code).

Runtime: ~7–8 minutes on a single workstation at the default mesh
resolution (``pixel_size = 4.0`` nm, ``Nv ≈ 31k`` vertices). Halve the
runtime by setting ``pixel_size = 6.0`` at the cost of slightly smeared
domain walls.

Outputs (saved to ``examples/output/``):
  ``tdbg_low_twist_relaxation.png``  — V_GSFE + |Δu| panels
  ``tdbg_low_twist_relaxation.npz``  — relaxed state
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from moire_metrology import (
    GRAPHENE_BILAYER_GRAPHENE_BILAYER,
    HexagonalLattice,
    MoireGeometry,
    PeriodicPairConstraint,
    RelaxationSolver,
    SolverConfig,
    generate_hex_periodic_mesh,
    identify_hex_periodic_boundary,
)
from moire_metrology.discretization import PinnedConstraints

OUTDIR = Path(__file__).parent / "output"
OUTDIR.mkdir(exist_ok=True)

# --- Problem parameters ----------------------------------------------------
THETA_DEG = 0.02      # twist angle
PIXEL = 4.0           # element size in nm (4 nm → Nv ≈ 31k at θ=0.02°)


def build_constraints(mesh, info, n_lay: int = 2):
    """Bottom-pinned + top-corner-pinned + top-edge periodicity."""
    Nv = mesh.n_vertices
    n_full = 2 * n_lay * Nv
    pinned: set[int] = set()
    for v in range(Nv):
        pinned.add(1 * Nv + v)
        pinned.add(n_lay * Nv + 1 * Nv + v)
    for v in info["corners"]:
        pinned.add(0 * Nv + int(v))
        pinned.add(n_lay * Nv + 0 * Nv + int(v))
    pinned_idx = np.array(sorted(pinned), dtype=np.int64)
    free_idx = np.setdiff1d(np.arange(n_full), pinned_idx, assume_unique=True)
    pc = PinnedConstraints(
        free_indices=free_idx, pinned_indices=pinned_idx,
        pinned_values=np.zeros(len(pinned_idx)),
        n_free=len(free_idx), n_full=n_full,
    )

    corner_set = {int(c) for c in info["corners"]}
    mean_cs = []
    for pair in info["pairs"]:
        src = pair["src_indices"]
        dst = pair["dst_indices"]
        keep = ~(np.isin(src, list(corner_set))
                 | np.isin(dst, list(corner_set)))
        pairs_arr = np.column_stack([src[keep], dst[keep]])
        mean_cs.append(PeriodicPairConstraint(layer_idx=0, pairs=pairs_arr))
    return pc, mean_cs


def filter_sliver_triangles(mesh, pixel: float) -> None:
    """Drop near-colinear (degenerate) triangles from the Delaunay output."""
    pts = mesh.points.T
    tri = mesh.triangles
    A = pts[tri[:, 0]]
    B = pts[tri[:, 1]]
    C = pts[tri[:, 2]]
    det = (B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) \
        - (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0])
    bad = np.abs(det) < 1e-6 * pixel**2
    if bad.any():
        print(f"  filtered {bad.sum()} sliver triangles (of {len(tri)})")
        mesh.triangles = tri[~bad]


def render(result, theta_deg: float, save_path: Path) -> None:
    """Two-panel V_GSFE + |Δu| over a 2×2 tile of the cell."""
    mesh = result.mesh
    pts, tri = mesh.points, mesh.triangles
    Nv = mesh.n_vertices
    V1, V2 = mesh.V1, mesh.V2

    ux1, ux2 = result.displacement_x1[0], result.displacement_x2[0]
    uy1, uy2 = result.displacement_y1[0], result.displacement_y2[0]
    du = np.sqrt((ux1 - ux2) ** 2 + (uy1 - uy2) ** 2)

    XY, F_g, F_du, T = [], [], [], []
    for i in range(2):
        for j in range(2):
            off = i * V1 + j * V2
            idx = len(XY)
            XY.append(np.column_stack([pts[0] + off[0], pts[1] + off[1]]))
            F_g.append(result.gsfe_map)
            F_du.append(du)
            T.append(tri + idx * Nv)
    triang = mtri.Triangulation(
        np.vstack(XY)[:, 0], np.vstack(XY)[:, 1], np.vstack(T),
    )

    vmax_gsfe = float(np.percentile(result.gsfe_map, 99.5))
    vmax_du = float(np.percentile(du, 99.0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    im0 = axes[0].tripcolor(
        triang, np.concatenate(F_g), shading="gouraud",
        cmap="magma", vmin=0, vmax=vmax_gsfe,
    )
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x [nm]")
    axes[0].set_ylabel("y [nm]")
    axes[0].set_title(f"V_GSFE  [meV / nm²]   (vmax = {vmax_gsfe:.1f})")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].tripcolor(
        triang, np.concatenate(F_du), shading="gouraud",
        cmap="magma", vmin=0, vmax=vmax_du,
    )
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("x [nm]")
    axes[1].set_ylabel("y [nm]")
    axes[1].set_title(f"|Δu|  [nm]   (vmax = {vmax_du:.2f})")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"TDBG-DFTD2 at θ = {theta_deg}°: 2DW relaxation\n"
        f"E = {result.total_energy:.4e} meV   "
        f"(nit = {result.optimizer_result.nit})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    print(f"   wrote {save_path}")


def main() -> None:
    matplotlib.use("Agg")
    ifc = GRAPHENE_BILAYER_GRAPHENE_BILAYER
    print(f"== TDBG-DFTD2 low-twist relaxation (θ = {THETA_DEG}°) ==")
    print(f"   interface: {ifc.name}")
    print(f"   K = {ifc.bottom.bulk_modulus}, "
          f"G = {ifc.bottom.shear_modulus}")
    print(f"   GSFE c0..c5 = {ifc.gsfe_coeffs}\n")

    lat = HexagonalLattice(alpha=ifc.bottom.lattice_constant)
    geom = MoireGeometry(lat, theta_twist=THETA_DEG, delta=0.0)
    mesh = generate_hex_periodic_mesh(geom, pixel_size=PIXEL)
    filter_sliver_triangles(mesh, PIXEL)
    info = identify_hex_periodic_boundary(mesh)
    print(f"   hex Wigner-Seitz cell: λ = {np.linalg.norm(geom.V1):.1f} nm, "
          f"Nv = {mesh.n_vertices}, Nt = {mesh.n_triangles}\n")

    pc, mean_cs = build_constraints(mesh, info)
    n_pair_rows = sum(c.n_rows for c in mean_cs)
    print(f"   constraints: {len(pc.pinned_indices)} pinned DOFs "
          f"(bottom layer + 6 top corners), "
          f"{n_pair_rows} periodic-pair rows\n")

    cfg = SolverConfig(
        method="two_phase", display=True, elastic_strain="green_lagrange",
        max_iter=200, max_iter_discover=200,
        gtol=1e-3, rtol=1e-4, etol=1e-7, etol_window=15,
    )
    t0 = perf_counter()
    result = RelaxationSolver(cfg).solve(
        moire_interface=ifc, theta_twist=THETA_DEG, delta=0.0,
        mesh=mesh, constraints=pc, mean_constraints=mean_cs,
    )
    elapsed = perf_counter() - t0
    print(f"\n   two_phase: E = {result.total_energy:.4e} meV, "
          f"nit = {result.optimizer_result.nit}, "
          f"t = {elapsed:.1f} s")

    render(result, THETA_DEG, OUTDIR / "tdbg_low_twist_relaxation.png")

    np.savez_compressed(
        OUTDIR / "tdbg_low_twist_relaxation.npz",
        theta_deg=THETA_DEG, pixel=PIXEL,
        points=mesh.points, triangles=mesh.triangles,
        V1=mesh.V1, V2=mesh.V2,
        total_energy=result.total_energy,
        solution_vector=result.solution_vector,
        gsfe_map=result.gsfe_map,
    )
    print(f"   wrote {OUTDIR / 'tdbg_low_twist_relaxation.npz'}")


if __name__ == "__main__":
    main()
