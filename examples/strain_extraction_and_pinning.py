"""Strain extraction and constrained finite-mesh relaxation.

A two-part example covering both spatially-uniform strain inversion
and the spatially-varying / point-pinned relaxation workflow:

Part A — Strain extraction (no relaxation)
==========================================
Reproduces (qualitatively) Figure 3c-d of:

    Halbertal, Shabani, Pasupathy, Basov, "Extracting the Strain Matrix
    and Twist Angle from the Moiré Superlattice in van der Waals
    Heterostructures", ACS Nano 16, 1471 (2022).
    https://doi.org/10.1021/acsnano.1c09789

Given a measured moiré superlattice characterized by the two moiré
lattice vectors (lengths λ₁, λ₂ and angles φ₁, φ₂) plus the atomic
lattice constants α₁, α₂, the inverse problem is to recover the twist
angle θ and the heterostrain tensor of the layers. The paper shows
that this is a one-parameter family of solutions parametrized by the
unknown substrate orientation φ₀; choosing φ₀ to minimize the
isotropic compression strain |ε_c|² gives a unique answer.

This part sweeps the angle Δφ ≡ φ₂ − φ₁ between the two moiré
vectors at fixed lengths λ₁ = λ₂, calls
``get_strain_minimize_compression`` for each, and plots the recovered
twist angle and strain components — paralleling the paper's Fig. 3c-d.
At Δφ = 60° the unstrained-symmetric solution is recovered exactly,
matching the analytic prediction λ = α / (2 sin θ/2). Away from 60°
the system has heterostrain, and the algorithm correctly factors the
twist out from the strain.

Part B — Constrained finite-mesh relaxation
===========================================
Demonstrates the spatially-varying-strain workflow enabled by the
finite-mesh + point-pinning support added in the non-periodic-mesh PR.

The setup is a small TBG flake on a finite triangular mesh; we pin a
few interior vertices to specific stacking configurations (AB / BA)
and let the relaxation fill in the rest by minimizing elastic + GSFE
energy. The pinned displacements are chosen so the moiré geometry at
those points matches a particular stacking — exactly the workflow you
would use to feed the output of an experimental stacking-site
identification (e.g. STM topograph of a TMD heterostructure) into a
post-relaxation calculation. Cf. Shabani, Halbertal et al.,
"Deep moiré potentials in twisted transition metal dichalcogenide
bilayers", arXiv:2008.07696, where the same workflow is used at much
larger scale to reconstruct the relaxed stacking-energy density of
real experimental images (their Fig. 1h).

Outputs (saved to examples/output/):
    strain_extraction_sweep.png      — recovered θ, ε_c, ε_s vs Δφ
    strain_pinned_relaxation.png     — relaxed stacking-energy map
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from moire_metrology import GRAPHENE, RelaxationSolver, SolverConfig
from moire_metrology.discretization import Discretization
from moire_metrology.lattice import HexagonalLattice, MoireGeometry
from moire_metrology.mesh import generate_finite_mesh
from moire_metrology.pinning import PinningMap
from moire_metrology.strain import get_strain_minimize_compression
from moire_metrology.strain.extraction import shear_strain_invariant


OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# --- Part A parameters --------------------------------------------------
# Match the paper's Fig. 3 setup: graphene on graphene (no lattice
# mismatch), λ₁ = λ₂ = 10 nm, φ₁ = 0°.
ALPHA = GRAPHENE.lattice_constant   # 0.247 nm
LAMBDA1 = LAMBDA2 = 10.0            # nm
PHI1_DEG = 0.0
DPHI_GRID = np.linspace(30.0, 90.0, 121)   # sweep Δφ around the unstrained 60°

# --- Part B parameters --------------------------------------------------
THETA_TWIST = 2.0       # degrees — moderate twist for fast relaxation
N_CELLS = 4             # finite mesh covers 4x4 moiré cells (~28 nm)
PIXEL_SIZE = 0.7        # nm


# =======================================================================
# Part A — Strain extraction sweep
# =======================================================================

def part_a_strain_sweep() -> None:
    """Sweep Δφ and plot recovered strain quantities, ACS Nano Fig. 3c-d."""
    print("=== Part A: strain extraction sweep ===")

    theta_recovered = np.zeros_like(DPHI_GRID)
    eps_c = np.zeros_like(DPHI_GRID)
    eps_s = np.zeros_like(DPHI_GRID)
    phi0 = np.zeros_like(DPHI_GRID)

    for k, dphi in enumerate(DPHI_GRID):
        result = get_strain_minimize_compression(
            alpha1=ALPHA, alpha2=ALPHA,
            lambda1=LAMBDA1, lambda2=LAMBDA2,
            phi1_deg=PHI1_DEG, phi2_deg=PHI1_DEG + dphi,
        )
        theta_recovered[k] = result.theta_twist
        eps_c[k] = result.eps_c
        # NOTE: we use the closed-form Eq 6 helper for ε_s rather than
        # result.eps_s. The latter is computed from the principal strains
        # of the recovered tensor, but currently has a φ₀-dependence bug
        # that the closed-form invariant does not have. See backlog
        # (get_strain_minimize_compression eps_s bug).
        eps_s[k] = shear_strain_invariant(
            alpha1=ALPHA, alpha2=ALPHA,
            lambda1=LAMBDA1, lambda2=LAMBDA2,
            phi1_deg=PHI1_DEG, phi2_deg=PHI1_DEG + dphi,
        )
        phi0[k] = result.phi0

    # Analytic prediction at Δφ = 60° (the unstrained-symmetric case):
    #   λ = α / (2 sin(θ/2))   ⇒   θ = 2 arcsin(α / (2λ))
    theta_unstrained_deg = 2.0 * np.degrees(np.arcsin(ALPHA / (2.0 * LAMBDA1)))

    # Sanity check: at Δφ = 60° the recovered twist should match analytics
    # and both strain components should vanish (the unstrained-symmetric
    # configuration).
    idx_60 = int(np.argmin(np.abs(DPHI_GRID - 60.0)))
    print(f"  λ₁ = λ₂ = {LAMBDA1} nm,  α = {ALPHA} nm")
    print(f"  Analytic θ at Δφ=60°:        {theta_unstrained_deg:.6f}°")
    print(f"  Recovered θ at Δφ=60°:       {theta_recovered[idx_60]:.6f}°")
    print(f"  ε_c (compression) at 60°:    {abs(eps_c[idx_60]):.2e}")
    print(f"  ε_s (shear, Eq 6) at 60°:    {abs(eps_s[idx_60]):.2e}")

    # ----- Plots ------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # (c) — recovered twist angle vs Δφ
    ax = axes[0]
    ax.plot(DPHI_GRID, theta_recovered, "b-", lw=1.5, label="recovered θ")
    ax.axhline(theta_unstrained_deg, color="k", ls="--", lw=0.8,
               label=f"λ = α/(2 sin θ/2)  ⇒  θ = {theta_unstrained_deg:.3f}°")
    ax.axvline(60.0, color="grey", ls=":", lw=0.7)
    ax.set_xlabel("Δφ = φ₂ − φ₁  (deg)")
    ax.set_ylabel("recovered twist angle θ  (deg)")
    ax.set_title("Strain extraction: recovered θ vs Δφ\n(ACS Nano Fig. 3c)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # (d) — recovered ε_c, ε_s vs Δφ
    ax = axes[1]
    ax.plot(DPHI_GRID, eps_c * 100, "k-", lw=1.5, label="ε_c (compression)")
    ax.plot(DPHI_GRID, eps_s * 100, "r-", lw=1.5, label="ε_s (shear)")
    ax.axvline(60.0, color="grey", ls=":", lw=0.7)
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("Δφ = φ₂ − φ₁  (deg)")
    ax.set_ylabel("strain (%)")
    ax.set_title(
        "Recovered strain vs Δφ  (ACS Nano Fig. 3d)\n"
        "ε_c minimized by choice of φ₀; ε_s is φ₀-invariant"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = OUT_DIR / "strain_extraction_sweep.png"
    fig.savefig(out_path, dpi=150)
    print(f"  Saved {out_path}")


# =======================================================================
# Part B — Constrained finite-mesh relaxation with stacking pins
# =======================================================================

def part_b_pinned_relaxation() -> None:
    """Build a finite mesh, pin natural moire AA sites, run the relaxation.

    The workflow:
      1. Build a finite-domain mesh covering several moire cells.
      2. Find the vertices that, in the unrelaxed (rigid) configuration,
         already correspond to high-symmetry AA stacking — these are
         the "natural AA sites" of the moire pattern at this twist.
      3. Pin those vertices to the AA stacking via PinningMap.
      4. Run a constrained relaxation. The pinned points act as
         topological anchors that prevent the system from drifting in
         a rigid translation while letting the rest of the displacement
         field find its energy minimum. The result is the same kind of
         relaxed AB/BA triangular domain network as in
         examples/bilayer_graphene_relaxation.py — but here it lives on
         a finite mesh, set up via the pinning API rather than via
         periodic boundary conditions.

    For the actual experimental workflow described in the Shabani /
    Halbertal Nature Physics paper (arXiv:2008.07696, Fig. 1h), the
    pinned positions would come from clicking on AA / XX' sites in an
    experimental STM topograph. Here we substitute synthetic AA sites
    derived from the geometry to make the example self-contained.
    """
    print("\n=== Part B: pinned finite-mesh relaxation ===")

    # 1. Build the finite-domain mesh.
    lattice = HexagonalLattice(alpha=GRAPHENE.lattice_constant)
    geometry = MoireGeometry(lattice, theta_twist=THETA_TWIST)
    mesh = generate_finite_mesh(
        geometry, n_cells=N_CELLS, pixel_size=PIXEL_SIZE,
    )
    print(f"  Mesh: {mesh.n_vertices} vertices, "
          f"{mesh.n_triangles} triangles, is_periodic={mesh.is_periodic}")
    print(f"  Moire wavelength: {geometry.wavelength:.2f} nm")

    # 2. Find vertices that sit at natural AA stacking in the unrelaxed
    # configuration. The geometry.stacking_phases(x, y) function returns
    # (v, w) which is (0, 0) modulo 2π at AA sites. We compute the
    # phase distance from AA at every vertex and select the vertices
    # within a small phase tolerance — those are our pin candidates.
    x = mesh.points[0]
    y = mesh.points[1]
    v, w = geometry.stacking_phases(x, y)
    # Wrap into [-π, π] so distance from 0 is well-defined
    v_wrapped = np.mod(v + np.pi, 2 * np.pi) - np.pi
    w_wrapped = np.mod(w + np.pi, 2 * np.pi) - np.pi
    phase_dist = np.sqrt(v_wrapped ** 2 + w_wrapped ** 2)
    # Pick a tolerance corresponding to ~10% of the AA→AB phase distance
    aa_threshold = 0.30 * (2 * np.pi / 3)  # AA→AB distance in v,w is 2π/3
    aa_candidates = np.where(phase_dist < aa_threshold)[0]
    print(f"  Natural AA candidate vertices: {len(aa_candidates)}")

    # Pin these vertices directly via the vertex-index API
    pins = PinningMap(mesh, geometry)
    pins.pin_vertices(aa_candidates, stacking="AA")
    n_pinned_v = len(pins.get_pinned_vertex_indices())
    print(f"  Pinned vertices total: {n_pinned_v}")

    # 3. Wire up the discretization and constraints.
    disc = Discretization(mesh, geometry)
    conv = disc.build_conversion_matrices(nlayer1=1, nlayer2=1)
    constraints = pins.build_constraints(conv, nlayer1=1, nlayer2=1)
    print(f"  Constraints: {constraints.n_free} free / "
          f"{len(constraints.pinned_indices)} pinned DOFs")

    # 4. Run the relaxation. Use L-BFGS-B (matrix-free, fast on small
    #    finite meshes; the new pseudo_dynamics iterative path would also
    #    work but is overkill for this size).
    cfg = SolverConfig(
        method="L-BFGS-B", pixel_size=PIXEL_SIZE, max_iter=300,
        gtol=1e-5, display=False,
    )
    print("  Running L-BFGS-B relaxation...")
    t0 = perf_counter()
    result = RelaxationSolver(cfg).solve(
        material1=GRAPHENE, material2=GRAPHENE, theta_twist=THETA_TWIST,
        mesh=mesh, constraints=constraints,
    )
    elapsed = perf_counter() - t0

    print(f"  Done in {elapsed:.2f} s")
    print(f"  Unrelaxed energy: {result.unrelaxed_energy:.2f} meV")
    print(f"  Relaxed energy:   {result.total_energy:.2f} meV")
    print(f"  Energy reduction: {100 * result.energy_reduction:.1f}%")

    # 5. Side-by-side plot: unrelaxed vs relaxed stacking-energy map.
    # The finite mesh is small enough that a per-vertex scatter is the
    # most legible rendering, and lets us overlay the pinned sites.
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Compute the unrelaxed stacking-energy map for comparison
    from moire_metrology.gsfe import GSFESurface
    gsfe = GSFESurface(GRAPHENE.gsfe_coeffs)
    v0, w0 = geometry.stacking_phases(x, y)
    g_unrelaxed = gsfe(v0, w0)
    g_relaxed = result.gsfe_map

    # Shared colour scale set by the unrelaxed maximum so the relaxation
    # is visible as a darkening of the plot
    vmax = float(np.percentile(g_unrelaxed, 95))

    pinned_indices = pins.get_pinned_vertex_indices()

    for ax, g, title in [
        (axes[0], g_unrelaxed, "Unrelaxed (rigid stacking)"),
        (axes[1], g_relaxed, f"Relaxed ({100*result.energy_reduction:.0f}% energy reduction)"),
    ]:
        sc = ax.scatter(
            mesh.points[0], mesh.points[1], c=g, cmap="magma",
            s=18, vmin=0.0, vmax=vmax,
        )
        # Overlay pinned vertices as small cyan circles
        ax.scatter(
            mesh.points[0, pinned_indices], mesh.points[1, pinned_indices],
            marker="o", s=80, edgecolor="cyan", facecolor="none",
            linewidth=1.2, label="pinned (AA)",
        )
        ax.set_aspect("equal")
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.set_title(title, fontsize=11)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        plt.colorbar(sc, ax=ax, label="GSFE (meV/nm²)")

    fig.suptitle(
        f"Pinned finite-mesh TBG relaxation, θ = {THETA_TWIST}°, "
        f"{N_CELLS}×{N_CELLS} moire cells\n"
        f"AA sites pinned via PinningMap; relaxation finds the AB/BA domain network",
        fontsize=12,
    )
    fig.tight_layout()
    out_path = OUT_DIR / "strain_pinned_relaxation.png"
    fig.savefig(out_path, dpi=150)
    print(f"  Saved {out_path}")


# =======================================================================

def main() -> None:
    part_a_strain_sweep()
    part_b_pinned_relaxation()


if __name__ == "__main__":
    main()
