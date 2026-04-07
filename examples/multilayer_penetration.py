"""Penetration of moire relaxation through a multi-layer graphene substrate.

Reproduces (qualitatively) the headline result of:
    Halbertal et al., "Multi-layered atomic relaxation in van der Waals
    heterostructures", arXiv:2206.06395 (2022/2023).

The paper's central claim is that moire-driven atomic relaxation is
NOT confined to the layers immediately at the twisted interface — it
propagates deep into the substrate. Fig. 2(d–f) of the paper shows
piezoresponse force microscopy maps of 2-, 5-, and 18-layer graphene
twisted at θ ≈ 0.14° on a graphite substrate, with the moire pattern
visibly resolved through all the layers above the buried interface.
The paper's continuum model explains this by the in-plane elastic
modulus being much larger than the GSFE coupling: the natural decay
length of the relaxation pattern through a graphene stack is many
layers, so a thin / few-layer stack relaxes essentially uniformly.

This example confirms that picture by performing a single relaxation
on a (1 twisted graphene) / (12 aligned graphene) stack at θ = 0.5°,
clamping the deepest substrate layer to zero displacement (fix_bottom)
to mimic a semi-infinite substrate. With 12 substrate layers the
bulk and the boundaries are well separated, and we can see:

 - The 11 free substrate layers all carry visibly the same moire
   pattern, with roughly uniform amplitude (~30 pm) — the bulk relaxes
   uniformly, exactly as the paper's argument predicts.
 - The interface-adjacent layer is not specially the largest; the
   substrate is "in equilibrium" with the twisted interface as a
   driving boundary condition.
 - The layer just above the clamp shows a small boundary bump, since
   it has to compromise between matching its bulk neighbour and
   matching the clamped layer at zero displacement.

Parameter choices differ from the paper for runtime reasons:

 - Twist angle 0.5° instead of 0.14°: per-iteration cost of the
   implicit solver scales steeply with mesh size, and the moire
   wavelength ~ 1/θ. At 0.5° the moire is ~28 nm and the calculation
   fits in a couple of minutes; at 0.14° it would take much longer.
   The qualitative bulk-uniform-relaxation picture is the same.
 - Substrate thickness 12 layers instead of 18, again for runtime.
 - Coarse mesh (~20×20) so each pseudo_dynamics step is fast.

The example uses two features that were specifically ported for this
demonstration:
 - method='pseudo_dynamics' on SolverConfig — the implicit theta-method
   solver from the paper's MATLAB code; the damped-Newton path stalls
   on multi-layer at low twist
 - fix_bottom=True on LayerStack.solve() — clamps the deepest substrate
   layer to zero displacement, matching the paper's `fix_external_layers`
   parameter

Outputs (saved to examples/output/):
    multilayer_penetration_relaxed.npz   — cached relaxed state
    multilayer_penetration_maps.png      — per-layer displacement maps
    multilayer_penetration_profile.png   — max |u| vs layer index
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from moire_metrology import GRAPHENE, SolverConfig
from moire_metrology.lattice import HexagonalLattice, MoireGeometry
from moire_metrology.mesh import MoireMesh
from moire_metrology.multilayer import LayerStack
from moire_metrology.plotting import plot_scalar_field
from moire_metrology.result import RelaxationResult


# --- Parameters ---------------------------------------------------------
THETA_TWIST = 0.5    # degrees. Low-twist regime; the paper uses 0.14°,
                     # 0.5° gives a ~28 nm moire that's tractable.
PIXEL_SIZE = 1.5     # nm.
MIN_MESH_POINTS = 20 # ~20×20 grid → ~400 vertices.
N_TOP = 1            # 1 twisted graphene layer.
N_BOTTOM = 12        # 12 aligned graphene layers in the substrate.
                     # The deepest is clamped; the other 11 relax.
                     # 12 is large enough to separate the bulk from
                     # the boundary effects in the per-layer profile.
MAX_ITER = 400       # pseudo_dynamics needs many small steps; on the
                     # tiny mesh chosen here this still finishes fast.

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)
CACHE_PATH = OUT_DIR / "multilayer_penetration_relaxed.npz"


def _load_cached_result() -> RelaxationResult | None:
    """Reconstruct a RelaxationResult from cache, or return None."""
    if not CACHE_PATH.exists():
        return None

    data = np.load(CACHE_PATH, allow_pickle=False)
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
        material1=GRAPHENE, material2=GRAPHENE,
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
    config = SolverConfig(
        method="pseudo_dynamics",   # the paper's algorithm; robust on
                                    # multi-layer at low twist where
                                    # Newton stalls
        pixel_size=PIXEL_SIZE,
        min_mesh_points=MIN_MESH_POINTS,
        max_iter=MAX_ITER,
        gtol=1e-3,                  # relative tolerance is fine here
        display=True,
    )
    stack = LayerStack(
        top=GRAPHENE, n_top=N_TOP,
        bottom=GRAPHENE, n_bottom=N_BOTTOM,
        theta_twist=THETA_TWIST,
    )
    print(stack.describe())
    print()
    # fix_bottom clamps the deepest substrate layer to zero displacement,
    # approximating a semi-infinite substrate (the conventional way to
    # avoid the spurious U-shape per-layer profile that comes from a
    # free-floating slab).
    return stack.solve(config, fix_bottom=True)


def _layer_max_u(displacement_x: np.ndarray, displacement_y: np.ndarray,
                 layer: int) -> float:
    """Maximum displacement magnitude at a given layer."""
    mag = np.sqrt(displacement_x[layer] ** 2 + displacement_y[layer] ** 2)
    return float(mag.max())


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
        print("No cache found — running multi-layer relaxation.")
        result = _solve()
        result.save(str(CACHE_PATH))
        print(f"Cached relaxed state → {CACHE_PATH}")
    else:
        print(f"Loaded cached relaxed state from {CACHE_PATH}")

    elapsed = perf_counter() - t0

    n_top = result.displacement_x1.shape[0]
    n_bot = result.displacement_x2.shape[0]
    print("\n--- Multi-layer relaxation summary ---")
    print(f"Stack:                {n_top} twisted top / {n_bot} aligned bottom")
    print(f"Twist angle:          {result.geometry.theta_twist:.3f}°")
    print(f"Moire wavelength:     {result.geometry.wavelength:.2f} nm")
    print(f"Mesh vertices:        {result.mesh.n_vertices}")
    print(f"Unrelaxed energy:     {result.unrelaxed_energy:.2f} meV")
    print(f"Relaxed energy:       {result.total_energy:.2f} meV")
    print(f"Energy reduction:     {100 * result.energy_reduction:.1f}%")
    print(f"Total wall time:      {elapsed:.1f} s")

    # --- Per-layer profile ---------------------------------------------
    # Layer ordering, going from the top of the heterostructure downward:
    #   stack 1 layer 0  (twisted graphene)               <- single layer here
    #   stack 2 layer 0  (substrate, adjacent to interface)
    #   stack 2 layer 1
    #   ...
    #   stack 2 layer (n_bot - 1)  (deepest, clamped to zero)
    layer_labels = [f"top {k}" for k in range(n_top)] + \
                   [f"bot {k}" for k in range(n_bot)]
    max_u = []
    for k in range(n_top):
        max_u.append(_layer_max_u(result.displacement_x1, result.displacement_y1, k))
    for k in range(n_bot):
        max_u.append(_layer_max_u(result.displacement_x2, result.displacement_y2, k))
    max_u_pm = np.array(max_u) * 1000  # nm → pm

    print("\n--- Per-layer max |u| ---")
    for label, mu in zip(layer_labels, max_u_pm):
        marker = "  (clamped)" if label == f"bot {n_bot - 1}" else ""
        print(f"  {label:7s}  max |u| = {mu:7.2f} pm{marker}")

    # --- Plot 1: penetration profile -----------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    layer_x = np.arange(len(max_u_pm))
    ax.plot(layer_x, max_u_pm, "o-", lw=2, ms=8, color="#c0392b")
    # Mark the interface between top flake and bottom flake.
    ax.axvline(n_top - 0.5, color="k", ls="--", lw=0.8, alpha=0.6,
               label="twisted interface")
    # Mark the clamped layer.
    ax.axvline(len(max_u_pm) - 1, color="b", ls=":", lw=1.0, alpha=0.7,
               label="clamped (fix_bottom)")
    ax.set_xticks(layer_x)
    ax.set_xticklabels(layer_labels, rotation=45, ha="right")
    ax.set_ylabel("max |u| in layer (pm)")
    ax.set_title(
        f"Per-layer relaxation amplitude through a {n_bot}-layer graphene substrate\n"
        f"θ = {THETA_TWIST}°, deepest substrate layer clamped"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "multilayer_penetration_profile.png", dpi=150)
    print(f"\nSaved {OUT_DIR / 'multilayer_penetration_profile.png'}")

    # --- Plot 2: per-layer displacement magnitude maps -----------------
    show = [(1, 0)] + [(2, k) for k in range(n_bot)]
    n_panels = len(show)
    ncols = min(5, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows),
                             squeeze=False)

    # Shared color scale, set by the strongest free layer.
    mags = {}
    vmax = 0.0
    for stack_idx, layer_idx in show:
        if stack_idx == 1:
            ux = result.displacement_x1[layer_idx]
            uy = result.displacement_y1[layer_idx]
        else:
            ux = result.displacement_x2[layer_idx]
            uy = result.displacement_y2[layer_idx]
        mag = np.sqrt(ux ** 2 + uy ** 2) * 1000  # pm
        mags[(stack_idx, layer_idx)] = mag
        vmax = max(vmax, float(np.percentile(mag, 99)))

    for panel, (stack_idx, layer_idx) in enumerate(show):
        ax = axes[panel // ncols, panel % ncols]
        plot_scalar_field(
            result.mesh, mags[(stack_idx, layer_idx)],
            ax=ax, n_tile=2, cmap="magma",
            vmin=0.0, vmax=vmax,
            colorbar=(panel == n_panels - 1),
        )
        if stack_idx == 1:
            title = "Twisted top layer"
        elif layer_idx == n_bot - 1:
            title = f"Bot layer {layer_idx} (clamped)"
        else:
            title = f"Bot layer {layer_idx}"
        ax.set_title(title, fontsize=11)

    # Hide unused axes
    for panel in range(n_panels, nrows * ncols):
        axes[panel // ncols, panel % ncols].set_visible(False)

    fig.suptitle(
        f"Per-layer relaxation magnitude (pm) — "
        f"{n_top} twisted / {n_bot} aligned graphene at θ = {THETA_TWIST}°",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "multilayer_penetration_maps.png", dpi=150)
    print(f"Saved {OUT_DIR / 'multilayer_penetration_maps.png'}")


if __name__ == "__main__":
    main()
