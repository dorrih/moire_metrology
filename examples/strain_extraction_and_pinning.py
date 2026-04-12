"""Strain extraction and spatially-varying constrained relaxation.

A two-part example. Part A is the inverse strain-extraction
calculation; Part B is the constrained finite-mesh relaxation
workflow that the package's `PinningMap` / `PinnedConstraints` and
`generate_finite_mesh` enable.

Part A -- Strain extraction sweep (no relaxation)
=================================================
Reproduces (qualitatively) Figure 3c-d of:

    Halbertal, Shabani, Pasupathy, Basov, "Extracting the Strain Matrix
    and Twist Angle from the Moire Superlattice in van der Waals
    Heterostructures", ACS Nano 16, 1471 (2022).
    https://doi.org/10.1021/acsnano.1c09789

Part B -- Imposed uniform heterostrain via pinned XX' points
============================================================
A finite-mesh constrained relaxation demonstrating how non-trivial
elongated domain-wall structures emerge when the pinned stacking
positions encode a uniform heterostrain.

Outputs (saved to examples/output/):
    strain_extraction_sweep.png            -- recovered theta, eps_c, eps_s vs dphi
    <slug>_pinned_heterostrain.png         -- Part B: uniform heterostrain

Usage examples::

    # Default (graphene sweep + MoSe2/WSe2 pinned relaxation)
    python strain_extraction_and_pinning.py

    # Custom Part B interface and parameters
    python strain_extraction_and_pinning.py --interface graphene-hbn --theta-twist 0.3

    # Skip Part A (strain sweep)
    python strain_extraction_and_pinning.py --skip-part-a

    # List available interfaces
    python strain_extraction_and_pinning.py --list-interfaces
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from moire_metrology import (
    GRAPHENE,
    RelaxationSolver, SolverConfig,
)
from moire_metrology.discretization import Discretization, PinnedConstraints
from moire_metrology.lattice import HexagonalLattice, MoireGeometry
from moire_metrology.mesh import generate_finite_mesh
from moire_metrology.strain import get_strain_minimize_compression

import _cli


# =======================================================================
# Part A -- Strain extraction sweep
# =======================================================================

def part_a_strain_sweep(out_dir: Path, lattice_constant: float) -> None:
    """Sweep dphi and plot recovered strain quantities, ACS Nano Fig. 3c-d."""
    print("=== Part A: strain extraction sweep ===")

    alpha = lattice_constant
    lambda1 = lambda2 = 10.0
    phi1_deg = 0.0
    dphi_grid = np.linspace(30.0, 90.0, 121)

    theta_recovered = np.zeros_like(dphi_grid)
    eps_c = np.zeros_like(dphi_grid)
    eps_s = np.zeros_like(dphi_grid)

    for k, dphi in enumerate(dphi_grid):
        result = get_strain_minimize_compression(
            alpha1=alpha, alpha2=alpha,
            lambda1=lambda1, lambda2=lambda2,
            phi1_deg=phi1_deg, phi2_deg=phi1_deg + dphi,
        )
        theta_recovered[k] = result.theta_twist
        eps_c[k] = result.eps_c
        eps_s[k] = abs(result.eps_s)

    theta_unstrained_deg = 2.0 * np.degrees(np.arcsin(alpha / (2.0 * lambda1)))

    idx_60 = int(np.argmin(np.abs(dphi_grid - 60.0)))
    print(f"  lambda1 = lambda2 = {lambda1} nm,  alpha = {alpha} nm")
    print(f"  Analytic theta at dphi=60:        {theta_unstrained_deg:.6f} deg")
    print(f"  Recovered theta at dphi=60:       {theta_recovered[idx_60]:.6f} deg")
    print(f"  eps_c (compression) at 60:        {abs(eps_c[idx_60]):.2e}")
    print(f"  eps_s (shear) at 60:              {abs(eps_s[idx_60]):.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(dphi_grid, theta_recovered, "b-", lw=1.5, label="recovered theta")
    ax.axhline(theta_unstrained_deg, color="k", ls="--", lw=0.8,
               label=f"theta = {theta_unstrained_deg:.3f} deg")
    ax.axvline(60.0, color="grey", ls=":", lw=0.7)
    ax.set_xlabel("dphi = phi2 - phi1  (deg)")
    ax.set_ylabel("recovered twist angle theta  (deg)")
    ax.set_title("Strain extraction: recovered theta vs dphi\n(ACS Nano Fig. 3c)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(dphi_grid, eps_c * 100, "k-", lw=1.5, label="eps_c (compression)")
    ax.plot(dphi_grid, eps_s * 100, "r-", lw=1.5, label="eps_s (shear)")
    ax.axvline(60.0, color="grey", ls=":", lw=0.7)
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("dphi = phi2 - phi1  (deg)")
    ax.set_ylabel("strain (%)")
    ax.set_title("Recovered strain vs dphi  (ACS Nano Fig. 3d)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "strain_extraction_sweep.png"
    fig.savefig(out_path, dpi=150)
    print(f"  Saved {out_path}")


# =======================================================================
# Pinning helpers
# =======================================================================

def _natural_high_symmetry_positions(
    geometry: MoireGeometry, mesh, *,
    sublattice: str = "AA", extent: int = 16,
) -> np.ndarray:
    """Enumerate natural high-symmetry positions of the unstrained moire
    that lie inside the bounding box of the supplied mesh."""
    offsets = {"AA": (0.0, 0.0), "AB": (1.0 / 3, 1.0 / 3), "BA": (2.0 / 3, 2.0 / 3)}
    if sublattice not in offsets:
        raise ValueError(f"sublattice must be one of {list(offsets)}")
    di, dj = offsets[sublattice]

    V1m = geometry.V1
    V2m = geometry.V2
    x_min, x_max = mesh.points[0].min(), mesh.points[0].max()
    y_min, y_max = mesh.points[1].min(), mesh.points[1].max()

    pts = []
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            s = i + di
            t = j + dj
            x = s * V1m[0] + t * V2m[0]
            y = s * V1m[1] + t * V2m[1]
            if x_min <= x <= x_max and y_min <= y <= y_max:
                pts.append((x, y))
    return np.asarray(pts)


def _nearest_vertex_indices(mesh, positions: np.ndarray) -> np.ndarray:
    """For each (x, y) in positions, return the index of the closest
    mesh vertex."""
    out = np.empty(len(positions), dtype=int)
    for k, (xs, ys) in enumerate(positions):
        d = (mesh.points[0] - xs) ** 2 + (mesh.points[1] - ys) ** 2
        out[k] = int(np.argmin(d))
    return out


def _build_displacement_pins(conv, vertex_indices: np.ndarray,
                             ux_targets: np.ndarray,
                             uy_targets: np.ndarray) -> PinnedConstraints:
    """Build a PinnedConstraints object that pins specific vertices to
    specific *relative* displacements (ux, uy)."""
    Nv = conv.n_vertices
    nlayer1 = conv.nlayer1
    nlayer2 = conv.nlayer2
    nlayers_total = nlayer1 + nlayer2
    n_full = conv.n_sol

    inner1 = nlayer1 - 1
    inner2 = nlayer1

    pinned_dofs: dict[int, float] = {}
    for vi, ux, uy in zip(vertex_indices, ux_targets, uy_targets):
        pinned_dofs[inner1 * Nv + int(vi)] = +ux / 2.0
        pinned_dofs[nlayers_total * Nv + inner1 * Nv + int(vi)] = +uy / 2.0
        pinned_dofs[inner2 * Nv + int(vi)] = -ux / 2.0
        pinned_dofs[nlayers_total * Nv + inner2 * Nv + int(vi)] = -uy / 2.0

    pinned_indices = np.array(sorted(pinned_dofs.keys()), dtype=int)
    pinned_values = np.array([pinned_dofs[i] for i in pinned_indices])
    free_indices = np.setdiff1d(np.arange(n_full), pinned_indices)

    return PinnedConstraints(
        free_indices=free_indices,
        pinned_indices=pinned_indices,
        pinned_values=pinned_values,
        n_free=len(free_indices),
        n_full=n_full,
    )


def _plot_pinned_relaxation(mesh, geometry, result, pinned_vertex_indices,
                            interface, *,
                            title: str, out_path: Path) -> None:
    """Side-by-side unrelaxed vs relaxed stacking-energy map with the
    pinned vertices marked."""
    from moire_metrology.gsfe import GSFESurface

    x = mesh.points[0]
    y = mesh.points[1]
    gsfe = GSFESurface(interface.gsfe_coeffs)
    v0, w0 = geometry.stacking_phases(x, y)
    g_unrelaxed = gsfe(v0, w0)
    g_relaxed = result.gsfe_map
    vmax = float(np.percentile(g_unrelaxed, 95))

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6))

    panels = [
        (axes[0], g_unrelaxed, "Unrelaxed (rigid stacking)", "magma",
         0.0, vmax, "GSFE (meV/nm^2)"),
        (axes[1], g_relaxed,
         f"Relaxed ({100 * result.energy_reduction:.0f}% energy reduction)",
         "magma", 0.0, vmax, "GSFE (meV/nm^2)"),
    ]

    for ax, field, label, cmap, vmin_p, vmax_p, cbar_label in panels:
        sc = ax.scatter(x, y, c=field, cmap=cmap, s=14,
                        vmin=vmin_p, vmax=vmax_p)
        ax.scatter(
            x[pinned_vertex_indices], y[pinned_vertex_indices],
            marker="o", s=70, edgecolor="cyan", facecolor="none",
            linewidth=1.2, label="pinned vertices",
        )
        ax.set_aspect("equal")
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.set_title(label, fontsize=11)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        plt.colorbar(sc, ax=ax, label=cbar_label)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Saved {out_path}")


# =======================================================================
# Part B -- Imposed uniform heterostrain via pinned AA points
# =======================================================================

def part_b_uniform_heterostrain(interface, theta, pixel_size,
                                heterostrain, n_cells,
                                out_dir: Path) -> None:
    """Pin natural XX' points to a uniform-heterostrain displacement field."""
    print(f"\n=== Part B: imposed uniform heterostrain ({interface.name}) ===")

    _cli.print_interface_info(interface)

    lattice = HexagonalLattice(alpha=interface.bottom.lattice_constant)
    geometry = MoireGeometry(
        lattice, theta_twist=theta,
        delta=interface.top.lattice_constant / interface.bottom.lattice_constant - 1.0,
    )
    mesh = generate_finite_mesh(
        geometry, n_cells=n_cells, pixel_size=pixel_size,
    )
    print(f"  theta = {theta} deg,  "
          f"intrinsic mismatch delta = {geometry.delta * 100:.3f}%")
    print(f"  Moire wavelength: {geometry.wavelength:.2f} nm")
    print(f"  Mesh: {mesh.n_vertices} vertices, "
          f"{mesh.n_triangles} triangles  ({n_cells}x{n_cells} moire cells)")

    aa_positions = np.concatenate([
        _natural_high_symmetry_positions(geometry, mesh, sublattice=s)
        for s in ("AA", "AB", "BA")
    ])
    r_c = np.array([mesh.points[0].mean(), mesh.points[1].mean()])
    print(f"  Natural high-symmetry points inside mesh "
          f"(XX'+MX'+MM'): {len(aa_positions)}")

    eps = np.array([[heterostrain, 0.0],
                    [0.0,          0.0]])

    dr = aa_positions - r_c
    u_targets = (eps @ dr.T).T
    ux_targets = u_targets[:, 0]
    uy_targets = u_targets[:, 1]
    print(f"  Heterostrain: {100 * heterostrain:.1f}% uniaxial along x")
    print(f"  Max |u_target| at pinned points: "
          f"{float(np.linalg.norm(u_targets, axis=1).max()) * 1000:.2f} pm")

    vertex_indices = _nearest_vertex_indices(mesh, aa_positions)

    disc = Discretization(mesh, geometry)
    conv = disc.build_conversion_matrices(nlayer1=1, nlayer2=1)
    constraints = _build_displacement_pins(
        conv, vertex_indices, ux_targets, uy_targets,
    )
    print(f"  Pinned DOFs: {len(constraints.pinned_indices)} of {constraints.n_full}")

    cfg = SolverConfig(
        method="L-BFGS-B", pixel_size=pixel_size, max_iter=400,
        gtol=1e-5, display=False,
    )
    print("  Running L-BFGS-B relaxation...")
    t0 = perf_counter()
    result = RelaxationSolver(cfg).solve(
        moire_interface=interface, theta_twist=theta,
        mesh=mesh, constraints=constraints,
    )
    print(f"  Done in {perf_counter() - t0:.2f} s")
    print(f"  Energy reduction: {100 * result.energy_reduction:.1f}%")

    slug = _cli.slugify(interface.name)
    _plot_pinned_relaxation(
        mesh, geometry, result, vertex_indices, interface,
        title=(f"Pinned {interface.name} relaxation, "
               f"{100 * heterostrain:.1f}% uniaxial heterostrain "
               f"(theta = {theta} deg)\n"
               f"XX' points pinned to u(r) = eps * (r - r_c)"),
        out_path=out_dir / f"{slug}_pinned_heterostrain.png",
    )


# =======================================================================

def main() -> None:
    parser = _cli.argparse.ArgumentParser(
        description=__doc__,
        formatter_class=_cli.argparse.RawDescriptionHelpFormatter,
    )
    _cli.add_interface_arg(parser, default="mose2-wse2-h")
    parser.add_argument(
        "--theta-twist", type=float, default=0.5, metavar="DEG",
        help="Twist angle for Part B (degrees). Default: %(default)s",
    )
    parser.add_argument(
        "--pixel-size", type=float, default=3.0, metavar="NM",
        help="Mesh pixel size for Part B (nm). Default: %(default)s",
    )
    parser.add_argument(
        "--heterostrain", type=float, default=0.01,
        help="Uniaxial heterostrain fraction for Part B. Default: %(default)s (1%%)",
    )
    parser.add_argument(
        "--n-cells", type=int, default=5,
        help="Number of moire cells per direction for Part B. Default: %(default)s",
    )
    parser.add_argument(
        "--lattice-constant", type=float, default=GRAPHENE.lattice_constant,
        metavar="NM",
        help="Lattice constant for Part A sweep (nm). Default: %(default)s (graphene)",
    )
    parser.add_argument(
        "--skip-part-a", action="store_true",
        help="Skip Part A (strain extraction sweep).",
    )
    parser.add_argument(
        "--skip-part-b", action="store_true",
        help="Skip Part B (pinned heterostrain relaxation).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, metavar="DIR",
        help="Output directory. Default: examples/output/",
    )
    args = parser.parse_args()
    _cli.handle_list_interfaces(args)

    out_dir = _cli.get_output_dir(args)

    if not args.skip_part_a:
        part_a_strain_sweep(out_dir, args.lattice_constant)

    if not args.skip_part_b:
        interface = _cli.resolve_interface(args.interface)
        part_b_uniform_heterostrain(
            interface=interface,
            theta=args.theta_twist,
            pixel_size=args.pixel_size,
            heterostrain=args.heterostrain,
            n_cells=args.n_cells,
            out_dir=out_dir,
        )


if __name__ == "__main__":
    main()
