"""Twisted bilayer relaxation.

A single configurable script for bilayer (two-layer) moire relaxation.
Replaces the former separate scripts for graphene, hBN, and TMD
heterostructures — the only differences were default parameters.

Common recipes (matching the previous standalone scripts)::

    # Twisted bilayer graphene (TBG) — small-twist triangular domains
    python bilayer_relaxation.py

    # Graphene on hBN — pure-mismatch moire at zero twist
    python bilayer_relaxation.py --preset hbn

    # H-stacked MoSe2/WSe2 — deep moire potential, three-stacking minima
    python bilayer_relaxation.py --preset tmd

    # Arbitrary interface from a TOML file
    python bilayer_relaxation.py --interface my_interface.toml --theta-twist 0.5

    # List available interfaces
    python bilayer_relaxation.py --list-interfaces

Presets
-------
Each ``--preset`` sets sensible defaults for a well-known system.  Any
explicit CLI flag overrides the preset value, so ``--preset tmd
--theta-twist 3.0`` runs the TMD interface at 3 deg instead of 1.5 deg.

    ========== ================ ===== ========== ======= ======== =====
    preset     interface        theta pixel_size method  max_iter gtol
    ========== ================ ===== ========== ======= ======== =====
    graphene   graphene         0.2   1.0        newton  60       1e-6
    hbn        graphene-hbn     0.0   0.5        L-BFGS-B 300    1e-4
    tmd        mose2-wse2-h     1.5   0.5        L-BFGS-B 300    1e-4
    ========== ================ ===== ========== ======= ======== =====

Outputs (saved to examples/output/):
    <slug>_relaxed.npz    — cached relaxed state (see --force)
    <slug>_stacking.png   — GSFE / stacking energy density
    <slug>_elastic.png    — Elastic energy density
    <slug>_twist.png      — Local twist angle map

The relaxed state is cached to a .npz file on first run and reused on
subsequent runs, so iterating on the plots is fast. Pass ``--force`` to
rebuild the cache from scratch.
"""

from __future__ import annotations

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from moire_metrology import RelaxationSolver
from moire_metrology.lattice import HexagonalLattice, MoireGeometry
from moire_metrology.mesh import MoireMesh
from moire_metrology.result import RelaxationResult

import _cli


# ── Presets ────────────────────────────────────────────────────────────

PRESETS = {
    "graphene": dict(
        interface="graphene", theta_twist=0.2, pixel_size=1.0,
        method="newton", max_iter=60, gtol=1e-6,
    ),
    "hbn": dict(
        interface="graphene-hbn", theta_twist=0.0, pixel_size=0.5,
        method="L-BFGS-B", max_iter=300, gtol=1e-4,
    ),
    "tmd": dict(
        interface="mose2-wse2-h", theta_twist=1.5, pixel_size=0.5,
        method="L-BFGS-B", max_iter=300, gtol=1e-4,
    ),
}
DEFAULT_PRESET = "graphene"


# ── Cache layer ───────────────────────────────────────────────────────

def _load_cached_result(cache_path, interface):
    """Reconstruct a RelaxationResult from the cache, or return None."""
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


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    preset_names = ", ".join(PRESETS)
    parser = _cli.argparse.ArgumentParser(
        description=__doc__,
        formatter_class=_cli.argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--preset", choices=list(PRESETS), default=None,
        help=(
            f"Load a named preset ({preset_names}). "
            "Explicit flags override preset values. "
            f"Default when no flags given: {DEFAULT_PRESET}"
        ),
    )
    # We add common args with graphene defaults; the preset logic
    # below overrides anything the user didn't explicitly set.
    defaults = PRESETS[DEFAULT_PRESET]
    _cli.add_common_args(
        parser,
        default_interface=defaults["interface"],
        default_theta=defaults["theta_twist"],
        default_pixel_size=defaults["pixel_size"],
        default_method=defaults["method"],
        default_max_iter=defaults["max_iter"],
        default_gtol=defaults["gtol"],
    )
    args = parser.parse_args()
    _cli.handle_list_interfaces(args)

    # Apply preset: for each key, if the user didn't explicitly pass it
    # (i.e. it's still at the default), override with the preset value.
    preset = PRESETS.get(args.preset or DEFAULT_PRESET, {})
    for key, val in preset.items():
        attr = key.replace("-", "_")
        if getattr(args, attr) == defaults.get(key):
            setattr(args, attr, val)

    interface = _cli.resolve_interface(args.interface)
    slug = _cli.slugify(interface.name)
    out_dir = _cli.get_output_dir(args)
    cache_path = out_dir / f"{slug}_relaxed.npz"
    theta = args.theta_twist

    _cli.print_interface_info(interface)

    t0 = perf_counter()

    result = None if args.force else _load_cached_result(cache_path, interface)
    if result is None:
        print("No cache found — running relaxation from scratch.\n")
        config = _cli.build_solver_config(args)
        solver = RelaxationSolver(config)
        print(f"=== {interface.name}, theta = {theta} deg ===\n")
        result = solver.solve(
            moire_interface=interface,
            theta_twist=theta,
        )
        result.save(str(cache_path))
        print(f"Cached relaxed state -> {cache_path}")
    else:
        print(f"Loaded cached relaxed state from {cache_path}")

    elapsed = perf_counter() - t0

    # --- Scalar summary ------------------------------------------------
    lam = result.geometry.wavelength
    dE = result.unrelaxed_energy - result.total_energy
    frac = result.energy_reduction
    summary_lines = [
        "--- Relaxation summary ---",
        f"Interface:            {interface.name}",
        f"Twist angle:          {theta} deg",
        f"Moire wavelength:     {lam:.2f} nm",
        f"Mesh vertices:        {result.mesh.n_vertices}",
        f"Unrelaxed energy:     {result.unrelaxed_energy:.2f} meV",
        f"Relaxed energy:       {result.total_energy:.2f} meV",
        f"Energy reduction:     {dE:.2f} meV  ({100 * frac:.1f}%)",
        f"Total wall time:      {elapsed:.1f} s",
    ]
    print("\n" + "\n".join(summary_lines))
    summary_path = out_dir / f"{slug}_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")

    if args.no_plots:
        return

    # --- Plots ---------------------------------------------------------
    # Adaptive tiling: small moire cells (hBN-like, lambda < 20 nm) get
    # 3x3 tiling to show periodicity; larger cells get 2x2.
    n_tile = 3 if lam < 20 else 2

    # (a) GSFE / stacking energy
    fig, ax = plt.subplots(figsize=(7, 6))
    vmax_g = float(np.percentile(result.gsfe_map, 93))
    result.plot_stacking(ax=ax, n_tile=n_tile, vmin=0.0, vmax=vmax_g)
    ax.set_title(
        f"{interface.name} stacking energy, theta = {theta} deg "
        f"({n_tile}x{n_tile} tiling)"
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"{slug}_stacking.png", dpi=150)
    print(f"Saved {out_dir / f'{slug}_stacking.png'}")

    # (b) Elastic energy density, top layer
    fig, ax = plt.subplots(figsize=(7, 6))
    elastic = result.elastic_map1[0]
    vmax_e = float(np.percentile(elastic, 99))
    result.plot_elastic_energy(stack=1, layer=0, ax=ax, n_tile=n_tile,
                               vmin=0.0, vmax=vmax_e)
    ax.set_title(
        f"{interface.name} elastic energy, top layer, theta = {theta} deg"
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"{slug}_elastic.png", dpi=150)
    print(f"Saved {out_dir / f'{slug}_elastic.png'}")

    # (c) Local twist angle
    fig, ax = plt.subplots(figsize=(7, 6))
    lt = result.local_twist(stack=1, layer=0)
    delta = float(np.percentile(np.abs(lt - theta), 99))
    result.plot_local_twist(stack=1, layer=0, ax=ax, n_tile=n_tile,
                            vmin=theta - delta,
                            vmax=theta + delta)
    ax.set_title(
        f"{interface.name} local twist, top layer, theta = {theta} deg"
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"{slug}_twist.png", dpi=150)
    print(f"Saved {out_dir / f'{slug}_twist.png'}")


if __name__ == "__main__":
    main()
