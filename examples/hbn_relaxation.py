"""Graphene on hBN heterointerface relaxation.

Demonstrates the moire-metrology relaxation workflow for a *heterobilayer*
where the moire pattern is driven entirely by the intrinsic lattice mismatch
between the two materials, even at zero twist.

Why this is qualitatively different from twisted bilayer graphene
----------------------------------------------------------------
- **Lattice mismatch dominates.** Graphene's lattice constant is 0.247 nm
  and hBN's is 0.251 nm, a δ ≈ 1.6% mismatch. The intrinsic moire wavelength
  at θ = 0 is λ = α / δ ≈ 15 nm — vastly smaller than the ~70 nm wavelength
  of TBG at θ = 0.2°. The relaxation problem is therefore much smaller and
  faster than the small-twist TBG hero case.
- **GSFE has a single minimum.** Unlike the AB / BA degeneracy of TBG, the
  graphene/hBN heterointerface has only one preferred stacking, so the
  relaxed pattern shows hexagonal domains around that single minimum
  rather than the AB/BA triangular network of TBG.
- **Broken inversion symmetry.** The G/hBN GSFE has nonzero c4, c5
  coefficients (it is not centrosymmetric), so the local stacking-energy
  landscape is intrinsically chiral.

References
----------
- Zhou et al. PRB 92, 155438 (2015), Table III, "G/BN" column — the
  source of the GSFE coefficients used in ``GRAPHENE_HBN_INTERFACE``.
- Carr et al. PRB 98, 224102 (2018), Table I — the source of the
  graphene K, G used in ``GRAPHENE``.
- Falin et al. Nat. Commun. 8, 15815 (2017) — the source of the
  hBN K, G used in ``HBN_AA``.

Outputs (saved to examples/output/):
    hbn_relaxation_stacking.png   — Stacking energy density (single minimum)
    hbn_relaxation_elastic.png    — Elastic energy density of the graphene
                                    layer (concentrates along domain walls)
    hbn_relaxation_summary.txt    — Scalar diagnostics

Runtime: ~10-30 seconds on a laptop. The mesh is small (~1000 vertices)
because the moire wavelength is intrinsically short.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from moire_metrology import (
    GRAPHENE,
    GRAPHENE_HBN_INTERFACE,
    HBN_AA,
    RelaxationSolver,
    SolverConfig,
)
from moire_metrology.result import RelaxationResult


# --- Parameters ---------------------------------------------------------
THETA_TWIST = 0.0     # degrees. Pure-mismatch case — moire is driven by
                      # δ alone, no twist needed. Try setting this to a
                      # small value like 0.3° to see how twist combines
                      # with the intrinsic mismatch.
PIXEL_SIZE = 0.5      # nm. The mismatch wavelength is ~15 nm so 0.5 nm
                      # pixels give ~30 grid points per moire-cell side
                      # (~1000 vertices per cell).
MAX_ITER = 300        # L-BFGS-B converges in 5-100 iterations depending
                      # on twist; the cap is just a safety net.

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)


def _solve() -> "RelaxationResult":
    # L-BFGS-B is preferred over Newton on this system because the
    # graphene/hBN GSFE is "stiff" enough that Newton's trust-region
    # damping inflates rapidly after the first few iterations and the
    # iteration log looks alarming, even though the actual converged
    # energy agrees with L-BFGS-B to within 0.01%. L-BFGS-B converges
    # in well under a second on this small mesh and reports a clean
    # convergence message.
    config = SolverConfig(
        method="L-BFGS-B",
        pixel_size=PIXEL_SIZE,
        max_iter=MAX_ITER,
        gtol=1e-4,
        display=True,
    )
    solver = RelaxationSolver(config)
    print(
        f"=== Graphene on hBN, θ = {THETA_TWIST}° "
        f"(pure-mismatch moire) ===\n"
    )
    return solver.solve(
        moire_interface=GRAPHENE_HBN_INTERFACE,
        theta_twist=THETA_TWIST,
    )


def _print_material_info() -> None:
    """Sanity-print the elastic moduli in literature N/m units to remind
    the reader that the bundled values trace back to specific papers."""
    K_g, G_g = GRAPHENE.moduli_n_per_m
    K_h, G_h = HBN_AA.moduli_n_per_m
    print("Material parameters (round-tripped to literature N/m units):")
    print(
        f"  Graphene: K = {K_g:6.1f} N/m, G = {G_g:6.1f} N/m  "
        f"(Carr et al. PRB 98 (2018), Table I)"
    )
    print(
        f"  hBN:      K = {K_h:6.1f} N/m, G = {G_h:6.1f} N/m  "
        f"(Falin et al. Nat. Commun. 8 (2017))"
    )
    print(
        f"  Lattice mismatch: δ = "
        f"{(GRAPHENE.lattice_constant / HBN_AA.lattice_constant - 1) * 100:.2f}%"
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
    (OUT_DIR / "hbn_relaxation_summary.txt").write_text("\n".join(summary_lines) + "\n")

    if args.no_plots:
        return

    # --- Plots ---------------------------------------------------------
    # The G/hBN moire is small (~15 nm wavelength) so a 3×3 tiling makes
    # the periodicity visually obvious without making the panel huge.

    # (a) Stacking energy — single-minimum hexagonal pattern
    fig, ax = plt.subplots(figsize=(7, 6))
    vmax_g = float(np.percentile(result.gsfe_map, 95))
    result.plot_stacking(ax=ax, n_tile=3, vmin=0.0, vmax=vmax_g)
    ax.set_title(
        f"Graphene/hBN stacking energy, θ = {THETA_TWIST}° (3×3 tiling)\n"
        f"Single-minimum hexagonal pattern; vmax clipped to 95th percentile"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hbn_relaxation_stacking.png", dpi=150)
    print(f"Saved {OUT_DIR / 'hbn_relaxation_stacking.png'}")

    # (b) Elastic energy density of the graphene (top) layer — concentrates
    # along the domain walls between low-energy stacking regions.
    fig, ax = plt.subplots(figsize=(7, 6))
    elastic = result.elastic_map1[0]
    vmax_e = float(np.percentile(elastic, 99))
    result.plot_elastic_energy(stack=1, layer=0, ax=ax, n_tile=3,
                               vmin=0.0, vmax=vmax_e)
    ax.set_title(
        f"Graphene/hBN elastic energy, graphene layer, θ = {THETA_TWIST}° "
        f"(3×3 tiling)"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hbn_relaxation_elastic.png", dpi=150)
    print(f"Saved {OUT_DIR / 'hbn_relaxation_elastic.png'}")


if __name__ == "__main__":
    main()
