"""Twisted bilayer graphene relaxation at small twist angle.

Reproduces the hallmark relaxation pattern of twisted bilayer graphene (TBG)
in the low twist-angle regime: triangular AB/BA domains separated by narrow
soliton (SDW) networks meeting at AA sites.

Reference
---------
Halbertal et al., "Moiré metrology of energy landscapes in van der Waals
heterostructures", Nat. Commun. 12, 242 (2021), arXiv:2008.04835.

In Fig. 1e of the paper, the authors show the relaxed stacking energy density
for TBG using the GSFE of Carr et al. (Ref. 31 in the paper). This example
uses the same GSFE (bundled as the `GRAPHENE` material in this package) at a
slightly larger twist angle than the paper's minimal-twist limit (θ < 0.1°)
to keep runtime in the 2-3 minute range on a laptop. The qualitative picture
— triangular AB/BA domains connected by SDWs at AA sites — is the same.

Outputs (saved to examples/output/):
    bilayer_graphene_relaxed.npz    — cached relaxed state (see --force)
    bilayer_graphene_stacking.png   — GSFE map (paper Fig. 1e analogue)
    bilayer_graphene_elastic.png    — Elastic energy density
    bilayer_graphene_twist.png      — Local twist angle map

The relaxed state is cached to a .npz file on first run and reused on
subsequent runs, so iterating on the plots is fast. Pass `--force` to
rebuild the cache from scratch.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from moire_metrology import GRAPHENE_GRAPHENE, RelaxationSolver, SolverConfig
from moire_metrology.lattice import HexagonalLattice, MoireGeometry
from moire_metrology.mesh import MoireMesh
from moire_metrology.result import RelaxationResult


# --- Parameters ---------------------------------------------------------
THETA_TWIST = 0.2  # degrees. Small-twist regime; below ~0.1° the moire cell
                   # grows as ~1/θ and relaxation cost scales as 1/θ², so 0.2°
                   # is a pragmatic compromise for a runnable example.
PIXEL_SIZE = 1.0   # nm. Mesh resolution; the moire wavelength at 0.2° is
                   # ~70 nm, so ~70 grid points per side → ~5k vertices. The
                   # relaxed soliton domain walls in TBG are ~5 nm wide at
                   # this twist, so 1 nm pixels resolve them with ~5 points.

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)
CACHE_PATH = OUT_DIR / "bilayer_graphene_relaxed.npz"


def _load_cached_result() -> RelaxationResult | None:
    """Reconstruct a RelaxationResult from the cache, or return None."""
    if not CACHE_PATH.exists():
        return None

    data = np.load(CACHE_PATH, allow_pickle=False)
    # Rebuild mesh (we only need the geometric primitives for plotting)
    mesh = MoireMesh(
        points=data["points"],
        triangles=data["triangles"],
        V1=data["V1"],
        V2=data["V2"],
        ns=int(data["ns"]),
        nt=int(data["nt"]),
        n_scale=int(data["n_scale"]),
    )
    lattice = HexagonalLattice(
        alpha=float(data["alpha"]),
        theta0=float(data["theta0"]),
    )
    geometry = MoireGeometry(
        lattice=lattice,
        theta_twist=float(data["theta_twist"]),
        delta=float(data["delta"]),
    )

    return RelaxationResult(
        mesh=mesh, geometry=geometry,
        moire_interface=GRAPHENE_GRAPHENE,
        top_interface=None,
        bottom_interface=None,
        displacement_x1=data["displacement_x1"],
        displacement_y1=data["displacement_y1"],
        displacement_x2=data["displacement_x2"],
        displacement_y2=data["displacement_y2"],
        total_energy=float(data["total_energy"]),
        unrelaxed_energy=float(data["unrelaxed_energy"]),
        gsfe_map=data["gsfe_map"],
        elastic_map1=data["elastic_map1"],
        elastic_map2=data["elastic_map2"],
        solution_vector=data["solution_vector"],
        optimizer_result=None,
    )


def _solve() -> RelaxationResult:
    # NOTE: gtol is set to an effectively-unreachable value on purpose.
    # At low twist angles the absolute gradient norm plateaus at O(1-10)
    # while the total energy is ~5×10^5 meV, so the default gtol=1e-6 is
    # unreachable (see task backlog: solver needs a relative convergence
    # criterion). Here we cap max_iter directly; by iter ~60 the energy
    # has plateaued to within ~0.01% of its asymptotic value, which is
    # more than enough to show the hallmark relaxation pattern.
    config = SolverConfig(
        method="newton",
        pixel_size=PIXEL_SIZE,
        max_iter=60,
        gtol=1e-6,
        display=True,
    )
    solver = RelaxationSolver(config)
    print(f"=== Twisted bilayer graphene, θ = {THETA_TWIST}° ===\n")
    result = solver.solve(
        moire_interface=GRAPHENE_GRAPHENE,
        theta_twist=THETA_TWIST,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-solve even if a cached .npz exists.",
    )
    args = parser.parse_args()

    t0 = perf_counter()

    result = None if args.force else _load_cached_result()
    if result is None:
        print("No cache found — running relaxation from scratch.")
        result = _solve()
        result.save(str(CACHE_PATH))
        print(f"Cached relaxed state → {CACHE_PATH}")
    else:
        print(f"Loaded cached relaxed state from {CACHE_PATH}")

    elapsed = perf_counter() - t0

    # --- Scalar summary ------------------------------------------------
    lam = result.geometry.wavelength
    dE = result.unrelaxed_energy - result.total_energy
    frac = result.energy_reduction
    print("\n--- Relaxation summary ---")
    print(f"Moire wavelength:     {lam:.2f} nm")
    print(f"Mesh vertices:        {result.mesh.n_vertices}")
    print(f"Unrelaxed energy:     {result.unrelaxed_energy:.2f} meV")
    print(f"Relaxed energy:       {result.total_energy:.2f} meV")
    print(f"Energy reduction:     {dE:.2f} meV  ({100 * frac:.1f}%)")
    print(f"Total wall time:      {elapsed:.1f} s")

    # --- Plots ---------------------------------------------------------
    # Percentile clipping is essential here: after relaxation the AA sites
    # become sharp isolated peaks over a vast AB/BA "sea" at the GSFE
    # minimum. A naive linear colormap is saturated by the AA peaks and
    # shows an almost-flat field. Clipping vmax to ~90-95th percentile
    # reveals the domain wall network and the triangular domain structure.

    # (a) GSFE / stacking energy — shows the AB/BA triangular domains
    fig, ax = plt.subplots(figsize=(7, 6))
    vmax_g = float(np.percentile(result.gsfe_map, 90))
    result.plot_stacking(ax=ax, n_tile=2, vmin=0.0, vmax=vmax_g)
    ax.set_title(
        f"TBG stacking energy density, θ = {THETA_TWIST}° (2×2 tiling)\n"
        f"vmax clipped to 90th percentile; AA peaks saturate above"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bilayer_graphene_stacking.png", dpi=150)
    print(f"Saved {OUT_DIR / 'bilayer_graphene_stacking.png'}")

    # (b) Elastic energy density in the top layer — shows domain wall strain
    fig, ax = plt.subplots(figsize=(7, 6))
    elastic = result.elastic_map1[0]
    vmax_e = float(np.percentile(elastic, 99))
    result.plot_elastic_energy(stack=1, layer=0, ax=ax, n_tile=2,
                               vmin=0.0, vmax=vmax_e)
    ax.set_title(f"TBG elastic energy, top layer, θ = {THETA_TWIST}° (2×2 tiling)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bilayer_graphene_elastic.png", dpi=150)
    print(f"Saved {OUT_DIR / 'bilayer_graphene_elastic.png'}")

    # (c) Local twist angle — AA sites show strong negative rotational
    # vortices; AB/BA domains sit near the global twist value.
    fig, ax = plt.subplots(figsize=(7, 6))
    lt = result.local_twist(stack=1, layer=0)
    # Symmetric clipping around global twist, dominated by AA dips
    delta = float(np.percentile(np.abs(lt - THETA_TWIST), 99))
    result.plot_local_twist(stack=1, layer=0, ax=ax, n_tile=2,
                            vmin=THETA_TWIST - delta,
                            vmax=THETA_TWIST + delta)
    ax.set_title(f"TBG local twist, top layer, θ = {THETA_TWIST}° (2×2 tiling)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bilayer_graphene_twist.png", dpi=150)
    print(f"Saved {OUT_DIR / 'bilayer_graphene_twist.png'}")


if __name__ == "__main__":
    main()
