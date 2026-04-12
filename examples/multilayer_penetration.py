"""Penetration of moire relaxation through a multi-layer substrate.

Reproduces (qualitatively) the headline result of:
    Halbertal et al., "Multi-layered atomic relaxation in van der Waals
    heterostructures", arXiv:2206.06395 (2022/2023).

The paper's central claim is that moire-driven atomic relaxation is
NOT confined to the layers immediately at the twisted interface — it
propagates deep into the substrate.

By default this runs with ``Graphene/Graphene`` at theta = 0.035 deg with
a 60/60 symmetric stack, but the interface, twist angle, and layer counts
can be changed via CLI arguments.  Only **homobilayer** interfaces are
supported (the internal flake stacking convention requires the same
material on both sides).

Outputs (saved to examples/output/):
    <slug>_multilayer_relaxed.npz     — cached relaxed state
    <slug>_multilayer_maps.png        — per-layer displacement maps
    <slug>_multilayer_profile.png     — peak epsilon vs layer index (log y)
    <slug>_multilayer_linecuts.png    — DW line cuts
    <slug>_multilayer_fwhm.png        — DW FWHM vs layer index

Usage examples::

    # Default (60/60 graphene at 0.035 deg)
    python multilayer_penetration.py

    # Smaller stack, faster
    python multilayer_penetration.py --n-top 5 --n-bottom 12 --theta-twist 0.5

    # hBN homobilayer
    python multilayer_penetration.py --interface hbn-aa --theta-twist 0.5 --n-top 5 --n-bottom 12

    # List available interfaces
    python multilayer_penetration.py --list-interfaces
"""

from __future__ import annotations

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from moire_metrology.lattice import HexagonalLattice, MoireGeometry
from moire_metrology.mesh import MoireMesh
from moire_metrology.multilayer import LayerStack
from moire_metrology.plotting import plot_scalar_field
from moire_metrology.result import RelaxationResult

import _cli


def _load_cached_result(cache_path, interface, n_top, n_bottom):
    """Reconstruct a RelaxationResult from cache, or return None."""
    if not cache_path.exists():
        return None

    data = np.load(cache_path, allow_pickle=False)
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
        moire_interface=interface,
        top_interface=None,
        bottom_interface=interface if n_bottom > 1 else None,
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


def _layer_peak_epsilon(elastic_map: np.ndarray, layer: int) -> float:
    """Peak elastic energy density in a given layer (meV/nm^2)."""
    return float(elastic_map[layer].max())


def main() -> None:
    parser = _cli.argparse.ArgumentParser(
        description=__doc__,
        formatter_class=_cli.argparse.RawDescriptionHelpFormatter,
    )
    _cli.add_common_args(
        parser,
        default_interface="graphene",
        default_theta=0.035,
        default_pixel_size=5.0,
        default_method="newton",
        default_max_iter=500,
        default_gtol=1e-3,
    )
    parser.add_argument(
        "--n-top", type=int, default=60, metavar="N",
        help="Number of layers in the top (twisted) flake. Default: %(default)s",
    )
    parser.add_argument(
        "--n-bottom", type=int, default=60, metavar="N",
        help="Number of layers in the bottom (aligned) flake. Default: %(default)s",
    )
    parser.add_argument(
        "--min-mesh-points", type=int, default=48, metavar="N",
        help="Floor on mesh grid resolution per direction. Default: %(default)s",
    )
    parser.add_argument(
        "--linear-solver", choices=["direct", "iterative"],
        default="iterative",
        help="Linear solver for Newton steps. Default: %(default)s",
    )
    parser.add_argument(
        "--linear-solver-tol", type=float, default=1e-4,
        help="Relative residual tolerance for iterative solver. Default: %(default)s",
    )
    parser.add_argument(
        "--linear-solver-maxiter", type=int, default=2000,
        help="Max iterations for iterative linear solver. Default: %(default)s",
    )
    args = parser.parse_args()
    _cli.handle_list_interfaces(args)

    interface = _cli.resolve_interface(args.interface)

    # Multilayer requires homobilayer + stacking function.
    _cli.require_homobilayer(interface, "multilayer_penetration.py")
    _cli.require_stacking_func(interface, "multilayer_penetration.py")

    slug = _cli.slugify(interface.name)
    out_dir = _cli.get_output_dir(args)
    cache_path = out_dir / f"{slug}_multilayer_relaxed.npz"
    theta = args.theta_twist
    n_top = args.n_top
    n_bottom = args.n_bottom

    _cli.print_interface_info(interface)

    t0 = perf_counter()

    result = None if args.force else _load_cached_result(
        cache_path, interface, n_top, n_bottom)
    if result is None:
        print("No cache found — running multi-layer relaxation.\n")
        config = _cli.build_solver_config(args)
        stack = LayerStack(
            moire_interface=interface,
            top_interface=interface if n_top > 1 else None,
            bottom_interface=interface if n_bottom > 1 else None,
            n_top=n_top,
            n_bottom=n_bottom,
            theta_twist=theta,
        )
        print(stack.describe())
        print()
        result = stack.solve(config, pin_mean=True)
        result.save(str(cache_path))
        print(f"Cached relaxed state -> {cache_path}")
    else:
        print(f"Loaded cached relaxed state from {cache_path}")

    elapsed = perf_counter() - t0

    n_top_actual = result.displacement_x1.shape[0]
    n_bot_actual = result.displacement_x2.shape[0]
    print("\n--- Multi-layer relaxation summary ---")
    print(f"Interface:            {interface.name}")
    print(f"Stack:                {n_top_actual} twisted top / {n_bot_actual} aligned bottom")
    print(f"Twist angle:          {result.geometry.theta_twist:.3f} deg")
    print(f"Moire wavelength:     {result.geometry.wavelength:.2f} nm")
    print(f"Mesh vertices:        {result.mesh.n_vertices}")
    print(f"Unrelaxed energy:     {result.unrelaxed_energy:.2f} meV")
    print(f"Relaxed energy:       {result.total_energy:.2f} meV")
    print(f"Energy reduction:     {100 * result.energy_reduction:.1f}%")
    print(f"Total wall time:      {elapsed:.1f} s")

    # --- Per-layer profile ---------------------------------------------
    layer_labels = [f"top {k}" for k in range(n_top_actual)] + \
                   [f"bot {k}" for k in range(n_bot_actual)]
    peak_eps = []
    for k in range(n_top_actual):
        peak_eps.append(_layer_peak_epsilon(result.elastic_map1, k))
    for k in range(n_bot_actual):
        peak_eps.append(_layer_peak_epsilon(result.elastic_map2, k))
    peak_eps = np.array(peak_eps)

    print("\n--- Per-layer peak elastic energy density (meV/nm^2) ---")
    for label, pe in zip(layer_labels, peak_eps):
        print(f"  {label:7s}  peak eps = {pe:10.4e}")

    if args.no_plots:
        return

    # --- Plot 1: penetration profile (log scale) -----------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    layer_x = np.arange(len(peak_eps))
    eps_floor = max(peak_eps.max() * 1e-8, 1e-12)
    peak_eps_plot = np.clip(peak_eps, eps_floor, None)
    ax.semilogy(layer_x, peak_eps_plot, "o-", lw=2, ms=6, color="#c0392b")
    ax.axvline(n_top_actual - 0.5, color="k", ls="--", lw=0.8, alpha=0.6,
               label="twisted interface")
    ax.set_xticks(layer_x[::max(1, len(layer_x) // 20)])
    ax.set_xticklabels(
        [layer_labels[i] for i in layer_x[::max(1, len(layer_x) // 20)]],
        rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("peak eps in layer (meV/nm^2)")
    ax.set_title(
        f"Penetration profile: peak elastic energy density vs layer\n"
        f"{interface.name}, theta = {theta} deg, {n_top_actual}/{n_bot_actual} stack"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_dir / f"{slug}_multilayer_profile.png", dpi=150)
    print(f"\nSaved {out_dir / f'{slug}_multilayer_profile.png'}")

    # --- Plot 2: per-layer elastic energy density maps -----------------
    from matplotlib.colors import LogNorm

    paper_d = [0, 2, 7, 21, n_top_actual - 1]
    top_slices = sorted({d for d in paper_d if 0 <= d < n_top_actual})
    paper_d_bot = [0, 2, 7, 21, n_bot_actual - 1]
    bot_slices = sorted({d for d in paper_d_bot if 0 <= d < n_bot_actual})
    show = [(1, n_top_actual - 1 - d) for d in top_slices] + \
           [(2, d) for d in bot_slices]
    n_panels = len(show)
    ncols = min(5, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows),
                             squeeze=False)

    emaps = {}
    for stack_idx, layer_idx in show:
        if stack_idx == 1:
            e = result.elastic_map1[layer_idx]
        else:
            e = result.elastic_map2[layer_idx]
        emaps[(stack_idx, layer_idx)] = e

    vmax = float(max(e.max() for e in emaps.values()))
    vmin = vmax * 1e-8
    norm = LogNorm(vmin=vmin, vmax=vmax)

    for panel, (stack_idx, layer_idx) in enumerate(show):
        ax = axes[panel // ncols, panel % ncols]
        e_clipped = np.clip(emaps[(stack_idx, layer_idx)], vmin, None)
        plot_scalar_field(
            result.mesh, e_clipped,
            ax=ax, n_tile=2, cmap="magma",
            norm=norm,
            colorbar=(panel == n_panels - 1),
        )
        if stack_idx == 1:
            d = n_top_actual - 1 - layer_idx
            title = f"Top k={d} (interface)" if d == 0 else f"Top k={d}"
        else:
            d = layer_idx
            if layer_idx == n_bot_actual - 1:
                title = f"Bot k={d} (clamped)"
            elif d == 0:
                title = f"Bot k={d} (interface)"
            else:
                title = f"Bot k={d}"
        ax.set_title(title, fontsize=11)

    for panel in range(n_panels, nrows * ncols):
        axes[panel // ncols, panel % ncols].set_visible(False)

    fig.suptitle(
        f"Per-layer elastic energy density (meV/nm^2, log scale) -- "
        f"{interface.name}, theta = {theta} deg, {n_top_actual}/{n_bot_actual} stack",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"{slug}_multilayer_maps.png", dpi=150)
    print(f"Saved {out_dir / f'{slug}_multilayer_maps.png'}")

    # --- Plot 3 & 4: DW line cuts and FWHM ----------------------------
    from scipy.interpolate import LinearNDInterpolator

    mesh = result.mesh
    points_2d = np.column_stack([mesh.points[0], mesh.points[1]])

    V1 = np.asarray(mesh.V1)
    V2 = np.asarray(mesh.V2)
    L_moire = float(np.hypot(V1[0], V1[1]))

    p_dw = 0.5 * (V1 + V2)
    diag = V1 + V2
    n_hat = diag / np.linalg.norm(diag)
    L_cut = L_moire / 2.0
    n_samples = 400
    t = np.linspace(-0.5, 0.5, n_samples) * L_cut
    line_x = p_dw[0] + t * n_hat[0]
    line_y = p_dw[1] + t * n_hat[1]
    line_s = t

    def _sample_layer(field_nv: np.ndarray) -> np.ndarray:
        interp = LinearNDInterpolator(points_2d, field_nv,
                                      fill_value=float(field_nv.min()))
        vals = interp(line_x, line_y)
        return np.clip(vals, 1e-12, None)

    samples_top = [
        _sample_layer(result.elastic_map1[n_top_actual - 1 - d])
        for d in range(n_top_actual)
    ]
    samples_bot = [
        _sample_layer(result.elastic_map2[d]) for d in range(n_bot_actual - 1)
    ]

    def _fwhm(y: np.ndarray, s: np.ndarray) -> float:
        i_peak = int(np.argmax(y))
        y_peak = y[i_peak]
        half = 0.5 * y_peak
        i = i_peak
        while i > 0 and y[i] > half:
            i -= 1
        if i == i_peak:
            return float("nan")
        s_left = np.interp(half, [y[i], y[i + 1]], [s[i], s[i + 1]])
        j = i_peak
        while j < len(y) - 1 and y[j] > half:
            j += 1
        if j == i_peak:
            return float("nan")
        s_right = np.interp(half, [y[j], y[j - 1]], [s[j], s[j - 1]])
        return float(s_right - s_left)

    fwhm_top = np.array([_fwhm(y, line_s) for y in samples_top])
    fwhm_bot = np.array([_fwhm(y, line_s) for y in samples_bot])

    fig_b, ax_b = plt.subplots(figsize=(7, 5))
    cmap_k = plt.get_cmap("viridis")
    for d, y in enumerate(samples_top):
        c = cmap_k(d / max(n_top_actual - 1, 1))
        ax_b.semilogy(line_s, y, color=c, lw=1.2,
                      label=f"top k={d}" if d in (0, n_top_actual // 2, n_top_actual - 1) else None)
    ax_b.axvline(0.0, color="k", ls=":", lw=0.8, alpha=0.5)
    ax_b.set_xlabel("s (nm)  --  perp. distance across DW")
    ax_b.set_ylabel("eps (meV/nm^2)")
    ax_b.set_title(
        f"Line cut perpendicular to a single DW\n"
        f"{interface.name}, theta = {theta} deg, top flake"
    )
    ax_b.grid(True, which="both", alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap=cmap_k,
                               norm=plt.Normalize(vmin=0, vmax=n_top_actual - 1))
    sm.set_array([])
    fig_b.colorbar(sm, ax=ax_b, label="layer index k (0 = interface)")
    fig_b.tight_layout()
    fig_b.savefig(out_dir / f"{slug}_multilayer_linecuts.png", dpi=150)
    print(f"Saved {out_dir / f'{slug}_multilayer_linecuts.png'}")

    fig_c, ax_c = plt.subplots(figsize=(7, 5))
    k_top = np.arange(n_top_actual)
    k_bot = np.arange(n_bot_actual - 1)
    ax_c.plot(k_top, fwhm_top, "o-", color="#c0392b", label="top flake")
    ax_c.plot(k_bot, fwhm_bot, "s-", color="#2980b9", label="bottom flake")
    ax_c.set_xlabel("distance from twisted interface (layer index k)")
    ax_c.set_ylabel("DW FWHM along line cut (nm)")
    ax_c.set_title(
        f"Domain-wall broadening through the stack\n"
        f"{interface.name}, theta = {theta} deg, {n_top_actual}/{n_bot_actual} stack"
    )
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(loc="best")
    fig_c.tight_layout()
    fig_c.savefig(out_dir / f"{slug}_multilayer_fwhm.png", dpi=150)
    print(f"Saved {out_dir / f'{slug}_multilayer_fwhm.png'}")

    print("\n--- Line-cut FWHM summary ---")
    print("  Layer k   FWHM_top (nm)   FWHM_bot (nm)")
    for k in range(max(n_top_actual, n_bot_actual - 1)):
        ft = f"{fwhm_top[k]:.2f}" if k < n_top_actual else "   -  "
        fb = f"{fwhm_bot[k]:.2f}" if k < n_bot_actual - 1 else "   -  "
        print(f"  k = {k:3d}    {ft:>8s}        {fb:>8s}")


if __name__ == "__main__":
    main()
