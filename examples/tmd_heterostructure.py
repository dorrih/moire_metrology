"""Twisted MoSe2/WSe2 (H-stacked) heterointerface relaxation.

Demonstrates the moire-metrology workflow for a transition-metal-dichalcogenide
(TMD) heterobilayer with a "deep moire potential" — GSFE coefficients an
order of magnitude larger than graphene's, and broken inversion symmetry
giving three distinct stacking minima per moire cell.

Why this is qualitatively different from twisted bilayer graphene
----------------------------------------------------------------
- **Deep GSFE.** The leading c0 coefficient for the H-stacked MoSe2/WSe2
  interface is 42.6 meV/uc — almost 7× larger than graphene's c0 = 6.832
  meV/uc. This means the interlayer registry energy is much stiffer
  relative to the elastic cost, so the relaxation is more aggressive
  and the domains are sharper.
- **Broken inversion symmetry.** The c4, c5 sin coefficients are nonzero
  (3.7 and 0.6 respectively, vs. exactly zero for graphene). The local
  stacking energy landscape has three distinct minima — XX', MX', and
  MM' — with different energies, rather than the AB/BA two-fold
  degeneracy of TBG. The MX' stacking is the global ground state.
- **Heterointerface, not homobilayer.** The two materials have different
  lattice constants (α_MoSe2 = 0.3288 nm, α_WSe2 = 0.3282 nm), giving
  an intrinsic δ ≈ 0.18% lattice mismatch. Combined with a small twist,
  this controls the moire wavelength.

References
----------
- Shabani, Halbertal et al. Nat. Phys. 17, 720 (2021), Methods section
  — the source of the H-stacked MoSe2/WSe2 GSFE coefficients used in
  ``MOSE2_WSE2_H_INTERFACE``, and an in-depth discussion of the deep
  moire potential and the three-stacking landscape.
- Halbertal et al. Nat. Commun. 12, 242 (2021), SI Table 1 — confirms
  the same K, G, and GSFE values used here.

Outputs (saved to examples/output/):
    tmd_heterostructure_stacking.png  — Stacking energy density showing
                                        the three distinct minima
    tmd_heterostructure_elastic.png   — Elastic energy density of the
                                        relaxed top layer
    tmd_heterostructure_summary.txt   — Scalar diagnostics

Runtime: ~30-60 seconds on a laptop. The mesh is moderate (~2000-3000
vertices) since the moire wavelength at the chosen twist is ~12 nm.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from moire_metrology import (
    MOSE2,
    MOSE2_WSE2_H_INTERFACE,
    WSE2,
    RelaxationSolver,
    SolverConfig,
)
from moire_metrology.result import RelaxationResult


# --- Parameters ---------------------------------------------------------
THETA_TWIST = 1.5     # degrees. With δ ≈ 0.18%, the moire wavelength at
                      # this twist is ~12 nm — small enough that the
                      # relaxation finishes in seconds, large enough
                      # that the three-fold stacking pattern is clearly
                      # visible in the plots.
PIXEL_SIZE = 0.5      # nm. ~24 grid points per moire-cell side at λ = 12 nm.
MAX_ITER = 300        # L-BFGS-B converges in well under 100 iterations
                      # on this system; the cap is just a safety net.

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)


def _solve() -> "RelaxationResult":
    # L-BFGS-B is preferred over Newton on this system. The deep MoSe2/WSe2
    # GSFE (c0 ≈ 7× larger than graphene) makes Newton's trust-region
    # damping inflate rapidly even though the actual converged energy
    # agrees with L-BFGS-B to within 0.01%. L-BFGS-B converges in well
    # under a second on this small mesh and reports a clean convergence
    # message.
    config = SolverConfig(
        method="L-BFGS-B",
        pixel_size=PIXEL_SIZE,
        max_iter=MAX_ITER,
        gtol=1e-4,
        display=True,
    )
    solver = RelaxationSolver(config)
    print(
        f"=== H-stacked MoSe2/WSe2 heterobilayer, θ = {THETA_TWIST}° ===\n"
    )
    return solver.solve(
        moire_interface=MOSE2_WSE2_H_INTERFACE,
        theta_twist=THETA_TWIST,
    )


def _print_material_info() -> None:
    """Sanity-print the elastic moduli in literature N/m units to remind
    the reader that the bundled values trace back to specific papers."""
    K_m, G_m = MOSE2.moduli_n_per_m
    K_w, G_w = WSE2.moduli_n_per_m
    print("Material parameters (round-tripped to literature N/m units):")
    print(
        f"  MoSe2: K = {K_m:6.1f} N/m, G = {G_m:6.1f} N/m  "
        f"(α = {MOSE2.lattice_constant} nm)"
    )
    print(
        f"  WSe2:  K = {K_w:6.1f} N/m, G = {G_w:6.1f} N/m  "
        f"(α = {WSE2.lattice_constant} nm)"
    )
    print(
        "  Source: Shabani, Halbertal et al. Nat. Phys. 17, 720 (2021), Methods section"
    )
    print(
        f"  Lattice mismatch: δ = "
        f"{(MOSE2.lattice_constant / WSE2.lattice_constant - 1) * 100:.3f}%"
    )
    print(
        "  GSFE c0..c5 (meV/uc): "
        f"{MOSE2_WSE2_H_INTERFACE.gsfe_coeffs}"
    )
    print(
        "  Note non-zero c4, c5 — broken inversion symmetry of the H-stacked\n"
        "  heterointerface, giving three distinct stacking minima (XX', MX', MM')."
    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip the plotting step (useful for headless smoke tests).",
    )
    args = parser.parse_args()

    _print_material_info()

    t0 = perf_counter()
    result = _solve()
    elapsed = perf_counter() - t0

    # --- Scalar summary ------------------------------------------------
    lam = result.geometry.wavelength
    dE = result.unrelaxed_energy - result.total_energy
    frac = result.energy_reduction
    summary_lines = [
        "--- Relaxation summary ---",
        f"Moire wavelength:     {lam:.2f} nm",
        f"Mesh vertices:        {result.mesh.n_vertices}",
        f"Unrelaxed energy:     {result.unrelaxed_energy:.2f} meV",
        f"Relaxed energy:       {result.total_energy:.2f} meV",
        f"Energy reduction:     {dE:.2f} meV  ({100 * frac:.1f}%)",
        f"Total wall time:      {elapsed:.1f} s",
    ]
    print("\n" + "\n".join(summary_lines))
    (OUT_DIR / "tmd_heterostructure_summary.txt").write_text(
        "\n".join(summary_lines) + "\n"
    )

    if args.no_plots:
        return

    # --- Plots ---------------------------------------------------------
    # The deep GSFE means the unrelaxed-vs-relaxed contrast is dramatic.
    # Use a 2×2 tiling so each panel shows enough moire cells to make the
    # three-stacking pattern visible.

    # (a) Stacking energy density — three distinct minima per moire cell.
    fig, ax = plt.subplots(figsize=(7, 6))
    # Don't clip the colormap as aggressively as for graphene — the deep
    # GSFE means the energy contrast is visually meaningful even at high
    # percentiles.
    vmax_g = float(np.percentile(result.gsfe_map, 98))
    result.plot_stacking(ax=ax, n_tile=2, vmin=0.0, vmax=vmax_g)
    ax.set_title(
        f"H-MoSe2/WSe2 stacking energy, θ = {THETA_TWIST}° (2×2 tiling)\n"
        f"Three-minimum pattern from broken inversion symmetry"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tmd_heterostructure_stacking.png", dpi=150)
    print(f"Saved {OUT_DIR / 'tmd_heterostructure_stacking.png'}")

    # (b) Elastic energy density of the top (MoSe2) layer.
    fig, ax = plt.subplots(figsize=(7, 6))
    elastic = result.elastic_map1[0]
    vmax_e = float(np.percentile(elastic, 99))
    result.plot_elastic_energy(stack=1, layer=0, ax=ax, n_tile=2,
                               vmin=0.0, vmax=vmax_e)
    ax.set_title(
        f"H-MoSe2/WSe2 elastic energy, MoSe2 layer, θ = {THETA_TWIST}° (2×2 tiling)"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tmd_heterostructure_elastic.png", dpi=150)
    print(f"Saved {OUT_DIR / 'tmd_heterostructure_elastic.png'}")


if __name__ == "__main__":
    main()
