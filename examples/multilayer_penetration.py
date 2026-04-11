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
    multilayer_penetration_profile.png   — peak ε vs layer index (log y)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from moire_metrology import GRAPHENE_GRAPHENE, SolverConfig
from moire_metrology.lattice import HexagonalLattice, MoireGeometry
from moire_metrology.mesh import MoireMesh
from moire_metrology.multilayer import LayerStack
from moire_metrology.plotting import plot_scalar_field
from moire_metrology.result import RelaxationResult


# --- Parameters ---------------------------------------------------------
THETA_TWIST = 0.035  # degrees — exactly matches Fig 4a of Halbertal
                     # et al. arXiv:2206.06395. Moire wavelength ~404 nm.
PIXEL_SIZE = 5.0     # nm. ~81×81 mesh on the 404 nm moire cell.
MIN_MESH_POINTS = 48
N_TOP = 60           # 60/60 exactly matches Fig 4a's stack size.
N_BOTTOM = 60        # Symmetric free-both-ends with single-vertex pin
                     # pattern away from the twisted interface on both
                     # sides. fix_bottom clamps the deepest layer.
MAX_ITER = 500       # CG-Newton takes ~2s/iter on this problem; giving
                     # it room because at very low twist the solver
                     # needs to break out of shallow metastable minima.

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
        moire_interface=GRAPHENE_GRAPHENE,
        top_interface=None,
        bottom_interface=GRAPHENE_GRAPHENE if N_BOTTOM > 1 else None,
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
        method="newton",            # modified-Hessian Newton (PR #16)
        linear_solver="iterative",  # preconditioned CG on the per-step
                                    # Newton system. Sparse LU fill-in is
                                    # crippling for this many layers at
                                    # low twist; CG with a Jacobi
                                    # preconditioner avoids the full LU.
        linear_solver_tol=1e-4,
        linear_solver_maxiter=2000,
        pixel_size=PIXEL_SIZE,
        min_mesh_points=MIN_MESH_POINTS,
        max_iter=MAX_ITER,
        gtol=1e-3,
        display=True,
    )
    stack = LayerStack(
        moire_interface=GRAPHENE_GRAPHENE,
        top_interface=GRAPHENE_GRAPHENE if N_TOP > 1 else None,
        bottom_interface=GRAPHENE_GRAPHENE if N_BOTTOM > 1 else None,
        n_top=N_TOP,
        n_bottom=N_BOTTOM,
        theta_twist=THETA_TWIST,
    )
    print(stack.describe())
    print()
    # Free-both-ends BC (matches Fig 4a of the paper): neither flake is
    # clamped. The 2D in-plane translation null space is broken by a
    # single-vertex (ux, uy) = 0 pin in the interface-adjacent layer.
    return stack.solve(config, pin_mean=True)


def _layer_peak_epsilon(elastic_map: np.ndarray, layer: int) -> float:
    """Peak elastic energy density in a given layer (meV/nm²)."""
    return float(elastic_map[layer].max())


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
    # Fig 4d of Halbertal et al. (2206.06395) shows the DECAY of the
    # elastic energy density amplitude through the stack — specifically
    # the peak ε^(j,k) vs layer index k, on a log-y axis. We use the
    # same metric here (peak ε per layer, in meV/nm²) rather than max
    # |u|, because ε is what the paper plots and it is more sensitive
    # to the sharpness of the relaxation pattern (ε ~ |∇u|² so two
    # decades of drop in ε ~ one decade in |u|).
    #
    # Layer ordering, going from the top of the heterostructure downward:
    #   stack 1 layer 0  (top flake topmost, free surface)
    #   stack 1 layer 1
    #   ...
    #   stack 1 layer (n_top - 1)  (top flake interface-adjacent)
    #   stack 2 layer 0            (bot flake interface-adjacent)
    #   ...
    #   stack 2 layer (n_bot - 1)  (bot flake deepest)
    layer_labels = [f"top {k}" for k in range(n_top)] + \
                   [f"bot {k}" for k in range(n_bot)]
    peak_eps = []
    for k in range(n_top):
        peak_eps.append(_layer_peak_epsilon(result.elastic_map1, k))
    for k in range(n_bot):
        peak_eps.append(_layer_peak_epsilon(result.elastic_map2, k))
    peak_eps = np.array(peak_eps)

    print("\n--- Per-layer peak elastic energy density ε (meV/nm²) ---")
    for label, pe in zip(layer_labels, peak_eps):
        print(f"  {label:7s}  peak ε = {pe:10.4e}")

    # --- Plot 1: penetration profile (log scale) -----------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    layer_x = np.arange(len(peak_eps))
    # Floor at a small positive value so log plot is well-defined.
    eps_floor = max(peak_eps.max() * 1e-8, 1e-12)
    peak_eps_plot = np.clip(peak_eps, eps_floor, None)
    ax.semilogy(layer_x, peak_eps_plot, "o-", lw=2, ms=6, color="#c0392b")
    ax.axvline(n_top - 0.5, color="k", ls="--", lw=0.8, alpha=0.6,
               label="twisted interface")
    ax.set_xticks(layer_x[::2])
    ax.set_xticklabels([layer_labels[i] for i in layer_x[::2]],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("peak ε in layer (meV/nm²)")
    ax.set_title(
        f"Fig-4d-style penetration profile: peak elastic energy density vs layer\n"
        f"θ = {THETA_TWIST}°, symmetric {n_top}/{n_bot} graphene"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "multilayer_penetration_profile.png", dpi=150)
    print(f"\nSaved {OUT_DIR / 'multilayer_penetration_profile.png'}")

    # --- Plot 2: per-layer elastic energy density (log scale) ----------
    # Matches the quantity plotted in Fig 4a of Halbertal et al.
    # (arXiv:2206.06395v2): layer-resolved elastic energy density
    # ε^(j,k)(r) in meV/nm², rendered on a shared log color scale so
    # that the penetration into the substrate is visible even when the
    # twisted top layer dominates the linear-scale amplitude.
    from matplotlib.colors import LogNorm

    # Representative layers matching Fig 4a of Halbertal et al.
    # (arXiv:2206.06395), which plots k = 1, 3, 8, 22, 60 where k=1
    # is the interface-adjacent layer. In our distance-from-interface
    # convention (d=0 is interface-adjacent), that maps to d = 0, 2,
    # 7, 21, n-1. The clustering near the interface is deliberate —
    # at very low twist the decay is so steep that a uniform sample
    # would push most panels below the color scale. For smaller
    # stacks we fall back to available layers.
    paper_d = [0, 2, 7, 21, n_top - 1]
    top_slices = sorted({d for d in paper_d if 0 <= d < n_top})
    paper_d_bot = [0, 2, 7, 21, n_bot - 1]
    bot_slices = sorted({d for d in paper_d_bot if 0 <= d < n_bot})
    # stack1 layer at distance d from interface = n_top - 1 - d
    show = [(1, n_top - 1 - d) for d in top_slices] + \
           [(2, d) for d in bot_slices]
    n_panels = len(show)
    ncols = min(5, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows),
                             squeeze=False)

    # Gather per-layer elastic-energy-density maps (meV/nm²).
    emaps = {}
    for stack_idx, layer_idx in show:
        if stack_idx == 1:
            e = result.elastic_map1[layer_idx]
        else:
            e = result.elastic_map2[layer_idx]
        emaps[(stack_idx, layer_idx)] = e

    # Shared log color scale spanning enough decades to show both the
    # interface layer (ε ~ 10¹ meV/nm² at very low twist) and the deep
    # bulk layers (Fig 4a of Halbertal et al. uses 10⁻⁷ to 10¹, 8
    # decades). We set vmax from the interface layer and vmin 8
    # decades below, matching the paper's convention; layers that have
    # decayed below 10⁻⁸ still contribute faint structure rather than
    # rendering uniformly black.
    vmax = float(max(e.max() for e in emaps.values()))
    vmin = vmax * 1e-8
    norm = LogNorm(vmin=vmin, vmax=vmax)

    for panel, (stack_idx, layer_idx) in enumerate(show):
        ax = axes[panel // ncols, panel % ncols]
        # Clip to vmin so log-plot handles zero/near-zero cells cleanly.
        e_clipped = np.clip(emaps[(stack_idx, layer_idx)], vmin, None)
        plot_scalar_field(
            result.mesh, e_clipped,
            ax=ax, n_tile=2, cmap="magma",
            norm=norm,
            colorbar=(panel == n_panels - 1),
        )
        if stack_idx == 1:
            # Distance from interface (interface at layer n_top-1)
            d = n_top - 1 - layer_idx
            title = f"Top k={d} (interface)" if d == 0 else f"Top k={d}"
        else:
            d = layer_idx  # interface at layer 0
            if layer_idx == n_bot - 1:
                title = f"Bot k={d} (clamped)"
            elif d == 0:
                title = f"Bot k={d} (interface)"
            else:
                title = f"Bot k={d}"
        ax.set_title(title, fontsize=11)

    # Hide unused axes
    for panel in range(n_panels, nrows * ncols):
        axes[panel // ncols, panel % ncols].set_visible(False)

    fig.suptitle(
        f"Per-layer elastic energy density (meV/nm², log scale) — "
        f"{n_top} twisted / {n_bot} aligned graphene at θ = {THETA_TWIST}°",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "multilayer_penetration_maps.png", dpi=150)
    print(f"Saved {OUT_DIR / 'multilayer_penetration_maps.png'}")

    # --- Plot 3 & 4: Fig-4b / Fig-4c diagnostics -----------------------
    # Fig 4b of arXiv:2206.06395 shows line cuts of the elastic energy
    # density ε^(j,k)(r) across a domain wall, one curve per layer k,
    # on a log y-axis. Fig 4c plots the FWHM of those line-cut peaks
    # vs layer index k, showing the DW broadening with distance from
    # the twisted interface.
    #
    # We take a horizontal line cut that spans one moire period along
    # the V1 primitive direction starting from the cell origin, and
    # sample the elastic energy density on that line for every layer
    # using matplotlib's triangular linear interpolator.
    # Use scipy's LinearNDInterpolator: robust to the mesh's periodic
    # wrap-around triangles in a way that matplotlib.tri is not.
    from scipy.interpolate import LinearNDInterpolator

    mesh = result.mesh
    points_2d = np.column_stack([mesh.points[0], mesh.points[1]])

    V1 = np.asarray(mesh.V1)
    V2 = np.asarray(mesh.V2)
    L_moire = float(np.hypot(V1[0], V1[1]))

    # Line-cut strategy: use a GEOMETRICALLY known DW midpoint instead
    # of detecting it from the relaxed fields. In the primitive cell
    # with vertices {0, V1, V2, V1+V2}, the two triangular stacking
    # domains meet at the diagonal from V1 to V2. The midpoint of that
    # diagonal is (V1+V2)/2, the geometric centre of the parallelogram,
    # which in stacking-phase space sits at (π, π) — an SP saddle of
    # the GSFE, i.e. a DW midpoint by symmetry.
    #
    # Putting the cut at (V1+V2)/2 has one other practical advantage:
    # it is the deepest interior point of the parallelogram, so a cut
    # of length ~L/2 in either direction stays entirely inside the
    # mesh (the alternative midpoint V1/2 lies on the bottom edge, so
    # half of that cut sampled outside the convex hull and hit the
    # LinearNDInterpolator fill value → the "flat one side" artifact).
    #
    # Cut direction: perpendicular to the diagonal DW tangent (V2-V1),
    # which in 2D is along (V1+V2)/|V1+V2|.
    p_dw = 0.5 * (V1 + V2)
    diag = V1 + V2
    n_hat = diag / np.linalg.norm(diag)
    L_cut = L_moire / 2.0
    n_samples = 400
    t = np.linspace(-0.5, 0.5, n_samples) * L_cut
    line_x = p_dw[0] + t * n_hat[0]
    line_y = p_dw[1] + t * n_hat[1]
    line_s = t  # s = 0 at the DW midpoint

    print("\n--- DW cut geometry (geometric midpoint) ---")
    print(f"  DW midpoint p = (V1+V2)/2 = ({p_dw[0]:.2f}, {p_dw[1]:.2f}) nm")
    print(f"  Cut direction n̂ = ({n_hat[0]:.3f}, {n_hat[1]:.3f}) "
          f"∥ V1+V2 (⊥ diagonal DW)")
    print(f"  Cut length = {L_cut:.2f} nm = L_moire/2")

    def _sample_layer(field_nv: np.ndarray) -> np.ndarray:
        interp = LinearNDInterpolator(points_2d, field_nv, fill_value=float(field_nv.min()))
        vals = interp(line_x, line_y)
        return np.clip(vals, 1e-12, None)

    # Sample every free layer on both sides. Order the curves from
    # the interface outward so the colormap encodes distance-from-interface.
    samples_top = [
        _sample_layer(result.elastic_map1[n_top - 1 - d])
        for d in range(n_top)
    ]
    samples_bot = [
        _sample_layer(result.elastic_map2[d]) for d in range(n_bot - 1)
    ]  # exclude the clamped layer

    def _fwhm(y: np.ndarray, s: np.ndarray) -> float:
        """FWHM of the dominant peak of y(s), in units of s."""
        i_peak = int(np.argmax(y))
        y_peak = y[i_peak]
        half = 0.5 * y_peak
        # Walk left
        i = i_peak
        while i > 0 and y[i] > half:
            i -= 1
        if i == i_peak:
            return float("nan")
        s_left = np.interp(half, [y[i], y[i + 1]], [s[i], s[i + 1]])
        # Walk right
        j = i_peak
        while j < len(y) - 1 and y[j] > half:
            j += 1
        if j == i_peak:
            return float("nan")
        s_right = np.interp(half, [y[j], y[j - 1]], [s[j], s[j - 1]])
        return float(s_right - s_left)

    fwhm_top = np.array([_fwhm(y, line_s) for y in samples_top])
    fwhm_bot = np.array([_fwhm(y, line_s) for y in samples_bot])

    # --- Fig-4b: line cuts, colored by layer index --------------------
    fig_b, ax_b = plt.subplots(figsize=(7, 5))
    cmap_k = plt.get_cmap("viridis")
    for d, y in enumerate(samples_top):
        c = cmap_k(d / max(n_top - 1, 1))
        ax_b.semilogy(line_s, y, color=c, lw=1.2,
                      label=f"top k={d}" if d in (0, n_top // 2, n_top - 1) else None)
    ax_b.axvline(0.0, color="k", ls=":", lw=0.8, alpha=0.5)
    ax_b.set_xlabel("s (nm)  —  perp. distance across DW")
    ax_b.set_ylabel("ε (meV/nm²)")
    ax_b.set_title(
        f"Fig-4b-style line cut perpendicular to a single DW\n"
        f"θ = {THETA_TWIST}°, top flake, colored by distance from interface"
    )
    ax_b.grid(True, which="both", alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap=cmap_k,
                               norm=plt.Normalize(vmin=0, vmax=n_top - 1))
    sm.set_array([])
    fig_b.colorbar(sm, ax=ax_b, label="layer index k (0 = interface)")
    fig_b.tight_layout()
    fig_b.savefig(OUT_DIR / "multilayer_penetration_linecuts.png", dpi=150)
    print(f"Saved {OUT_DIR / 'multilayer_penetration_linecuts.png'}")

    # --- Fig-4c: FWHM of the DW peak vs layer index k -----------------
    fig_c, ax_c = plt.subplots(figsize=(7, 5))
    k_top = np.arange(n_top)
    k_bot = np.arange(n_bot - 1)
    ax_c.plot(k_top, fwhm_top, "o-", color="#c0392b", label="top flake")
    ax_c.plot(k_bot, fwhm_bot, "s-", color="#2980b9", label="bottom flake")
    ax_c.set_xlabel("distance from twisted interface (layer index k)")
    ax_c.set_ylabel("DW FWHM along line cut (nm)")
    ax_c.set_title(
        f"Fig-4c-style domain-wall broadening through the stack\n"
        f"θ = {THETA_TWIST}°, symmetric {n_top}/{n_bot} graphene"
    )
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(loc="best")
    fig_c.tight_layout()
    fig_c.savefig(OUT_DIR / "multilayer_penetration_fwhm.png", dpi=150)
    print(f"Saved {OUT_DIR / 'multilayer_penetration_fwhm.png'}")

    print("\n--- Line-cut FWHM summary ---")
    print("  Layer k   FWHM_top (nm)   FWHM_bot (nm)")
    for k in range(max(n_top, n_bot - 1)):
        ft = f"{fwhm_top[k]:.2f}" if k < n_top else "   -  "
        fb = f"{fwhm_bot[k]:.2f}" if k < n_bot - 1 else "   -  "
        print(f"  k = {k:3d}    {ft:>8s}        {fb:>8s}")


if __name__ == "__main__":
    main()
