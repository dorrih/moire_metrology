"""End-to-end spatially-varying strain extraction + relaxation example.

Reproduces Fig. 1c-e of Halbertal, Shabani, Pasupathy & Basov, ACS Nano
16, 1471 (2022) on a real H-stacked MoSe2/WSe2 sample with non-uniform
twist angle and strain across the field of view, then **extends** the
workflow by feeding the recovered local strain field as the initial
condition for an atomic relaxation against ``MOSE2_WSE2_H_INTERFACE``.
The result is a continuous prediction of the equilibrium domain
pattern that minimizes the deep moiré potential GSFE under the
spatial constraints from the strain extraction.

Pipeline:

1. :class:`FringeSet` loads two families of integer-labeled polylines
   from a MATLAB ``.mat`` file (or any other source).
2. :meth:`FringeSet.fit_registry_fields` fits the two registry index
   polynomials ``I(x, y)`` and ``J(x, y)`` to all polyline points.
3. :func:`compute_strain_field` recovers
   ``(θ(r), S11(r), S12(r), S22(r))`` and the derived ``(ε_c, ε_s)``
   scalars from ``(I, J)`` via the eq. 9 inversion of the paper.
4. :func:`convex_hull_mask` zeroes the strain field outside the data
   support; the high-degree polynomial extrapolates wildly outside its
   training data and would otherwise produce nonsense displacements.
5. :func:`displacement_from_strain_field` integrates the local
   ``(δθ, S)`` field on the FEM mesh to produce the displacement IC
   ``u(r)`` in the relaxation framework's native language. With the
   strain field zeroed outside the hull, the integrated ``u(r)``
   smoothly relaxes to the average configuration there.
6. :class:`RelaxationSolver` runs the constrained relaxation against
   ``MOSE2_WSE2_H_INTERFACE`` from this IC, using the
   ``pseudo_dynamics`` (implicit theta-method) solver — recommended
   for stiff multi-layer / low twist cases.

Headline figure (saved to ``examples/output/``): a 2×2 panel with
``θ(x, y)``, ``ε_c(x, y)``, ``ε_s(x, y)``, and the relaxed stacking
energy density.

The relaxation step takes ~10 minutes. Its result is cached to
``examples/output/spatial_strain_relaxed.npz`` after the first
successful run; subsequent runs reload from the cache and finish in a
few seconds. Pass ``--force`` to re-solve from scratch.

References
----------
* Halbertal, Shabani, Pasupathy & Basov, "Extracting the strain matrix
  and twist angle from the moiré superlattice in van der Waals
  heterostructures", ACS Nano 16, 1471 (2022),
  doi:10.1021/acsnano.1c09789. Source of the strain extraction and the
  Fig 1 panels being reproduced.
* Shabani, Halbertal et al., Nat. Phys. 17, 720 (2021),
  doi:10.1038/s41567-021-01174-7. Source of the H-MoSe2/WSe2 GSFE used
  by ``MOSE2_WSE2_H_INTERFACE`` in the relaxation step.

Data file
---------
The example looks for ``docs_internal/Bilayer--183_1_pointsList.mat``
relative to the repo root. This file is gitignored (maintainer-only).
If absent, the example exits with a message; the API surface is the
same regardless of the data source — drop in any ``.mat`` produced by
the maintainer's polyline GUI, or build a :class:`FringeSet` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from moire_metrology import (
    RelaxationSolver,
    SolverConfig,
)
from moire_metrology.discretization import Discretization
from moire_metrology.lattice import HexagonalLattice, MoireGeometry
from moire_metrology.mesh import MoireMesh, generate_finite_mesh
from moire_metrology.pinning import PinningMap
from moire_metrology.plotting import plot_scalar_field
from moire_metrology.strain import (
    FringeSet,
    compute_strain_field,
    convex_hull_mask,
    displacement_from_strain_field,
)

import _cli


# --- Default parameters (overridable via CLI) ----------------------------

DEFAULT_DATA_PATH = (Path(__file__).parents[1]
                     / "docs_internal" / "Bilayer--183_1_pointsList.mat")
DEFAULT_INTERFACE = "mose2-wse2-h"
DEFAULT_POLY_DEGREE = 11
DEFAULT_PHI0_DEG = -65.6
DEFAULT_N_GRID = 100
DEFAULT_N_CELLS = 55
DEFAULT_PIXEL_SIZE = 2.0
DEFAULT_MAX_ITER = 70


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------


@dataclass
class _CachedResult:
    """Minimal stand-in for ``RelaxationResult`` carrying just the
    fields the headline figure needs.

    The shared ``RelaxationResult.save()`` doesn't preserve the finite-
    mesh ``is_periodic=False`` flag (it was written for periodic-mesh
    examples), so this example uses its own small cache layer.
    """
    mesh: MoireMesh
    gsfe_map: np.ndarray
    energy_reduction: float
    total_energy: float
    unrelaxed_energy: float


def _save_cache(result, mesh: MoireMesh, path: Path) -> None:
    np.savez_compressed(
        path,
        points=mesh.points,
        triangles=mesh.triangles,
        V1=mesh.V1,
        V2=mesh.V2,
        ns=mesh.ns,
        nt=mesh.nt,
        n_scale=mesh.n_scale,
        is_periodic=mesh.is_periodic,
        gsfe_map=result.gsfe_map,
        total_energy=result.total_energy,
        unrelaxed_energy=result.unrelaxed_energy,
        energy_reduction=result.energy_reduction,
    )


def _load_cache(path: Path) -> _CachedResult | None:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    mesh = MoireMesh(
        points=data["points"],
        triangles=data["triangles"],
        V1=data["V1"],
        V2=data["V2"],
        ns=int(data["ns"]),
        nt=int(data["nt"]),
        n_scale=int(data["n_scale"]),
        is_periodic=bool(data["is_periodic"]),
    )
    return _CachedResult(
        mesh=mesh,
        gsfe_map=data["gsfe_map"],
        energy_reduction=float(data["energy_reduction"]),
        total_energy=float(data["total_energy"]),
        unrelaxed_energy=float(data["unrelaxed_energy"]),
    )


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------


def _build_mesh_for_data(fringes: FringeSet, theta_avg_deg: float,
                         alpha1: float, alpha2: float,
                         n_cells: int, pixel_size: float):
    """Build a finite mesh covering the data extent at the average geometry."""
    delta_avg = alpha2 / alpha1 - 1.0
    lattice = HexagonalLattice(alpha=alpha1)
    geom = MoireGeometry(lattice, theta_twist=theta_avg_deg, delta=delta_avg)
    mesh = generate_finite_mesh(geom, n_cells=n_cells, pixel_size=pixel_size)

    xs = np.concatenate([f.x for f in fringes.fringes])
    ys = np.concatenate([f.y for f in fringes.fringes])
    shift_x = (
        (xs.min() + xs.max()) / 2
        - (mesh.points[0].min() + mesh.points[0].max()) / 2
    )
    shift_y = (
        (ys.min() + ys.max()) / 2
        - (mesh.points[1].min() + mesh.points[1].max()) / 2
    )
    mesh.points[0] += shift_x
    mesh.points[1] += shift_y
    return mesh, geom


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_headline(
    fringes: FringeSet,
    X: np.ndarray, Y: np.ndarray,
    strain: dict, mesh, result,
    out_path: Path,
    *,
    poly_degree: int = 11,
    phi0_deg: float = -65.6,
) -> None:
    """2x2 panel: strain extraction maps + relaxed stacking energy."""
    xs_I = np.concatenate([f.x for f in fringes.i_fringes])
    ys_I = np.concatenate([f.y for f in fringes.i_fringes])
    xs_J = np.concatenate([f.x for f in fringes.j_fringes])
    ys_J = np.concatenate([f.y for f in fringes.j_fringes])

    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    extent = [X.min(), X.max(), Y.min(), Y.max()]

    # ----- (top-left) recovered θ with sampled-points overlay -----
    ax = axes[0, 0]
    sc = ax.imshow(
        np.abs(strain["theta"]),
        extent=extent, origin="lower",
        cmap="inferno", vmin=0.4, vmax=1.9,
    )
    ax.scatter(xs_I, ys_I, s=1.5, c="cyan", alpha=0.4, label="I polyline pts")
    ax.scatter(xs_J, ys_J, s=1.5, c="lime", alpha=0.4, label="J polyline pts")
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title(
        f"recovered θ(x, y) — paper Fig 1c\n"
        f"polynomial degree {poly_degree}, phi0 = {phi0_deg:.1f} deg"
    )
    plt.colorbar(sc, ax=ax, label="θ (deg)", shrink=0.8)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.8)

    # ----- (top-right) recovered ε_c -----
    ax = axes[0, 1]
    sc = ax.imshow(
        strain["eps_c"] * 100,
        extent=extent, origin="lower",
        cmap="RdBu_r", vmin=-0.2, vmax=0.2,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
    ax.set_title(
        "recovered ε_c(x, y) — paper Fig 1d\n"
        f"mean = {np.nanmean(strain['eps_c']) * 100:.4f}%, "
        f"std = {np.nanstd(strain['eps_c']) * 100:.4f}%"
    )
    plt.colorbar(sc, ax=ax, label="ε_c (%)", shrink=0.8)

    # ----- (bottom-left) recovered ε_s -----
    ax = axes[1, 0]
    sc = ax.imshow(
        strain["eps_s"] * 100,
        extent=extent, origin="lower",
        cmap="inferno", vmin=0, vmax=0.8,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title(
        "recovered ε_s(x, y) — paper Fig 1e\n"
        f"mean = {np.nanmean(strain['eps_s']) * 100:.4f}%, "
        f"std = {np.nanstd(strain['eps_s']) * 100:.4f}%"
    )
    plt.colorbar(sc, ax=ax, label="ε_s (%)", shrink=0.8)

    # ----- (bottom-right) relaxed stacking energy density -----
    ax = axes[1, 1]
    if result is None:
        ax.text(0.5, 0.5, "(relaxation skipped)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_aspect("equal")
    else:
        g = result.gsfe_map
        vmin_g = float(np.nanpercentile(g, 5))
        vmax_g = float(np.nanpercentile(g, 95))
        plot_scalar_field(
            mesh, g, ax=ax, n_tile=1,
            cmap="magma", vmin=vmin_g, vmax=vmax_g,
        )
        ax.set_aspect("equal")
        ax.set_xlabel("x (nm)")
        ax.set_title(
            "RELAXED stacking energy density\n"
            f"{result.mesh.n_vertices} verts, "
            f"{100 * result.energy_reduction:.1f}% energy reduction"
        )

    pad = 20.0
    for ax in axes.flat:
        ax.set_xlim(X.min() - pad, X.max() + pad)
        ax.set_ylim(Y.min() - pad, Y.max() + pad)

    fig.suptitle(
        "Spatially-varying strain extraction + relaxation\n"
        "Top row + bottom-left reproduce Halbertal et al. ACS Nano 16, 1471 "
        "(2022) Fig 1c-e. Bottom-right is the relaxation prediction.",
        fontsize=11, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _cli.argparse.ArgumentParser(
        description=__doc__,
        formatter_class=_cli.argparse.RawDescriptionHelpFormatter,
    )
    _cli.add_interface_arg(parser, default=DEFAULT_INTERFACE)
    parser.add_argument(
        "--data-file", type=Path, default=DEFAULT_DATA_PATH, metavar="PATH",
        help="Path to the .mat polyline data file. Default: %(default)s",
    )
    parser.add_argument(
        "--poly-degree", type=int, default=DEFAULT_POLY_DEGREE,
        help="Polynomial degree for registry fits. Default: %(default)s",
    )
    parser.add_argument(
        "--phi0", type=float, default=DEFAULT_PHI0_DEG, metavar="DEG",
        help="Substrate orientation phi0 (degrees). Default: %(default)s",
    )
    parser.add_argument(
        "--n-grid", type=int, default=DEFAULT_N_GRID,
        help="Strain map grid resolution. Default: %(default)s",
    )
    parser.add_argument(
        "--n-cells", type=int, default=DEFAULT_N_CELLS,
        help="Mesh cell count for relaxation. Default: %(default)s",
    )
    parser.add_argument(
        "--pixel-size", type=float, default=DEFAULT_PIXEL_SIZE, metavar="NM",
        help="Mesh pixel size (nm). Default: %(default)s",
    )
    parser.add_argument(
        "--max-iter", type=int, default=DEFAULT_MAX_ITER,
        help="Max relaxation iterations. Default: %(default)s",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plotting (useful for headless smoke testing).",
    )
    parser.add_argument(
        "--no-relax", action="store_true",
        help="Skip the relaxation step (only do strain extraction).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-solve even if a cached relaxed state exists.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, metavar="DIR",
        help="Output directory. Default: examples/output/",
    )
    args = parser.parse_args()
    _cli.handle_list_interfaces(args)

    interface = _cli.resolve_interface(args.interface)
    ALPHA1 = interface.bottom.lattice_constant
    ALPHA2 = interface.top.lattice_constant
    poly_degree = args.poly_degree
    phi0_deg = args.phi0
    n_grid = args.n_grid
    n_cells = args.n_cells
    pixel_size = args.pixel_size
    max_iter = args.max_iter
    OUT_DIR = _cli.get_output_dir(args)
    slug = _cli.slugify(interface.name)
    CACHE_PATH = OUT_DIR / f"{slug}_spatial_relaxed.npz"

    _cli.print_interface_info(interface)

    # ----- Step 1: load polylines into a FringeSet -----
    data_path = args.data_file
    if not data_path.exists():
        print(
            f"Data file {data_path} not found.\n"
            "This example requires a .mat polyline data file. "
            "See the module docstring for the expected format."
        )
        return
    print(f"Loading polylines from {data_path}")
    fringes = FringeSet.from_matlab(data_path)
    print(f"  family 1 (I): {len(fringes.i_fringes)} polylines")
    print(f"  family 2 (J): {len(fringes.j_fringes)} polylines")

    # ----- Step 2: fit registry polynomials I(r), J(r) -----
    print(f"\nFitting registry polynomials of degree {poly_degree}...")
    I_field, J_field = fringes.fit_registry_fields(order=poly_degree)

    # ----- Step 3: compute strain field on a regular grid -----
    print("Computing strain field on a regular grid...")
    xs = np.concatenate([f.x for f in fringes.fringes])
    ys = np.concatenate([f.y for f in fringes.fringes])
    x_grid = np.linspace(xs.min(), xs.max(), n_grid)
    y_grid = np.linspace(ys.min(), ys.max(), n_grid)
    X, Y = np.meshgrid(x_grid, y_grid)

    inside_grid = convex_hull_mask(xs, ys, X, Y)

    strain = compute_strain_field(
        X, Y, I_field, J_field,
        alpha1=ALPHA1, alpha2=ALPHA2, phi0_deg=phi0_deg,
    )
    for key in ("theta", "eps_c", "eps_s"):
        strain[key] = np.where(inside_grid, strain[key], np.nan)

    theta = strain["theta"]
    eps_c = strain["eps_c"]
    eps_s = strain["eps_s"]
    print(
        f"  |θ|: range [{np.nanmin(np.abs(theta)):.3f}, "
        f"{np.nanmax(np.abs(theta)):.3f}], "
        f"mean = {np.nanmean(np.abs(theta)):.3f}"
    )
    print(
        f"  ε_c: mean = {np.nanmean(eps_c) * 100:.4f}%, "
        f"std = {np.nanstd(eps_c) * 100:.4f}%"
    )
    print(
        f"  ε_s: mean = {np.nanmean(eps_s) * 100:.4f}%, "
        f"std = {np.nanstd(eps_s) * 100:.4f}%"
    )

    result = None
    mesh = None
    elapsed = 0.0

    if not args.no_relax:
        # Try the cache first.
        cached = None if args.force else _load_cache(CACHE_PATH)
        if cached is not None:
            print(f"\nLoaded cached relaxed state from {CACHE_PATH}")
            print("(use --force to re-solve)")
            result = cached
            mesh = cached.mesh

    if not args.no_relax and result is None:
        # ----- Step 4: build mesh and FEM gradient operators -----
        theta_avg = float(np.nanmean(np.abs(theta)))
        print(f"\nBuilding finite mesh at theta_avg = {theta_avg:.3f} deg...")
        mesh, geom = _build_mesh_for_data(fringes, theta_avg,
                                           ALPHA1, ALPHA2, n_cells, pixel_size)
        disc = Discretization(mesh, geom)
        print(f"  Mesh: {mesh.n_vertices} verts, {mesh.n_triangles} triangles")
        print(f"  Average geometry: λ = {geom.wavelength:.2f} nm")

        # ----- Step 5: build IC by evaluating strain at mesh vertices,
        # masking to the data hull, and integrating the gradient field. -----
        print("Evaluating local strain field at mesh vertices...")
        vertex_strain = compute_strain_field(
            mesh.points[0], mesh.points[1], I_field, J_field,
            alpha1=ALPHA1, alpha2=ALPHA2, phi0_deg=phi0_deg,
        )
        # The package's compute_strain_field reports -θ relative to the
        # MoireGeometry convention, so use |θ| for the IC.
        theta_v = np.abs(vertex_strain["theta"])
        S11_v = vertex_strain["S11"]
        S12_v = vertex_strain["S12"]
        S22_v = vertex_strain["S22"]

        # Mask the strain field to the convex hull of the polyline data.
        # Outside the hull, set δθ = 0 and S = 0 — the gradient-integration
        # IC will then smoothly relax u to a constant there.
        mesh_inside = convex_hull_mask(xs, ys, mesh.points[0], mesh.points[1])
        theta_v = np.where(mesh_inside, theta_v, theta_avg)
        S11_v = np.where(mesh_inside, S11_v, 0.0)
        S12_v = np.where(mesh_inside, S12_v, 0.0)
        S22_v = np.where(mesh_inside, S22_v, 0.0)
        n_inside = int(mesh_inside.sum())
        print(f"  {n_inside}/{mesh.n_vertices} mesh vertices inside data hull")

        # ----- Step 6: find polyline intersections and pin them -----
        # Intersections of I-polylines and J-polylines are sites where
        # both registry indices are simultaneously integer — the true
        # AA stacking sites of the experimental moiré lattice (domain
        # wall junctions). Pinning these anchors the domain orientation
        # and phase to the experimental data.
        # Find ALL three stacking sub-lattice sites (AA, BA, AB) from
        # the polynomial registry fit. Pinning all three ensures the
        # IC (via Dirichlet-BC Poisson) already has the correct domain
        # pattern — BA and AB domains at low GSFE, with AA at the
        # domain-wall junctions. Without BA/AB pins, the Dirichlet-BC
        # interpolation between all-AA sites puts the whole mesh near
        # AA stacking (~63 meV/uc), which is the wrong starting point.
        print("Finding stacking sub-lattice sites from polynomial...")
        I_at_mesh = I_field(mesh.points[0], mesh.points[1])
        J_at_mesh = J_field(mesh.points[0], mesh.points[1])
        I_mod = I_at_mesh % 1.0
        J_mod = J_at_mesh % 1.0
        tol = 0.15

        def _frac_dist(mod_val, target):
            """Circular distance of (val mod 1) from target in [0,1)."""
            d = np.abs(mod_val - target)
            return np.minimum(d, 1.0 - d)

        # AA: I mod 1 ≈ 0, J mod 1 ≈ 0
        is_aa = mesh_inside & (_frac_dist(I_mod, 0.0) < tol) & (_frac_dist(J_mod, 0.0) < tol)
        # BA: I mod 1 ≈ 2/3, J mod 1 ≈ 2/3
        is_ba = mesh_inside & (_frac_dist(I_mod, 2/3) < tol) & (_frac_dist(J_mod, 2/3) < tol)
        # AB: I mod 1 ≈ 1/3, J mod 1 ≈ 1/3
        is_ab = mesh_inside & (_frac_dist(I_mod, 1/3) < tol) & (_frac_dist(J_mod, 1/3) < tol)

        pins = PinningMap(mesh, geom)
        pin_set: set[int] = set()
        for stacking, mask in [("AA", is_aa), ("BA", is_ba), ("AB", is_ab)]:
            verts = np.where(mask)[0]
            for vi in verts:
                pin_set.add(int(vi))
            pins.pin_vertices(verts, stacking=stacking)
            print(f"  {stacking}: {len(verts)} vertices")

        constraints = pins.build_constraints(
            disc.build_conversion_matrices(nlayer1=1, nlayer2=1),
        )
        print(
            f"  Total: {len(pin_set)} pinned mesh vertices, "
            f"{len(constraints.pinned_indices)} pinned DOFs / "
            f"{constraints.n_full}"
        )

        # Build the gradient-IC with ALL pin displacements as Dirichlet
        # BCs. With AA, BA, and AB pins, the Poisson interpolation now
        # produces the correct domain pattern in the IC (BA domains at
        # low GSFE, AA junctions at high GSFE).
        print("Building IC via gradient integration + Dirichlet BCs...")
        all_pin_verts = np.array(sorted(pin_set), dtype=int)
        # Look up the stacking each vertex was pinned to.
        dir_ux = np.empty(len(all_pin_verts))
        dir_uy = np.empty(len(all_pin_verts))
        for i, vi in enumerate(all_pin_verts):
            # Determine which stacking this vertex has.
            if is_aa[vi]:
                stk = "AA"
            elif is_ba[vi]:
                stk = "BA"
            else:
                stk = "AB"
            ux_v, uy_v = pins._compute_displacement_for_stacking(int(vi), stk)
            dir_ux[i] = ux_v
            dir_uy[i] = uy_v

        ux, uy = displacement_from_strain_field(
            disc,
            theta_deg=theta_v, theta_avg_deg=theta_avg,
            S11=S11_v, S12=S12_v, S22=S22_v,
            dirichlet_vertices=all_pin_verts,
            dirichlet_ux=dir_ux,
            dirichlet_uy=dir_uy,
        )
        print(
            f"  IC: max |u| = {float(np.abs(np.hypot(ux, uy)).max()):.3f} nm, "
            f"rms |u| = {float(np.sqrt((ux ** 2 + uy ** 2).mean())):.3f} nm"
        )

        # Pack into the solver's full DOF vector. For nlayer1 = nlayer2 = 1
        # the layout is [+ux/2, -ux/2, +uy/2, -uy/2] (eps=0.5 partition).
        Nv = mesh.n_vertices
        U_full = np.zeros(4 * Nv)
        U_full[0 * Nv:1 * Nv] = +ux / 2.0
        U_full[1 * Nv:2 * Nv] = -ux / 2.0
        U_full[2 * Nv:3 * Nv] = +uy / 2.0
        U_full[3 * Nv:4 * Nv] = -uy / 2.0
        U0_free = U_full[constraints.free_indices]

        # ----- Step 7: relaxation via pseudo_dynamics -----
        cfg = SolverConfig(
            method="newton",
            pixel_size=pixel_size,
            max_iter=max_iter,
            gtol=1e-4,
            display=True,
        )
        print(f"\nRunning relaxation (max {max_iter} iterations)...")
        t0 = perf_counter()
        result = RelaxationSolver(cfg).solve(
            moire_interface=interface,
            theta_twist=geom.theta_twist,
            mesh=mesh,
            constraints=constraints,
            initial_solution=U0_free,
        )
        elapsed = perf_counter() - t0
        print(
            f"  Done in {elapsed:.1f}s, "
            f"energy reduction = {100 * result.energy_reduction:.1f}%"
        )

        # Cache the relaxed state for fast plot iteration.
        _save_cache(result, mesh, CACHE_PATH)
        print(f"  Cached relaxed state → {CACHE_PATH}")

    # ----- Scalar summary -----
    summary_lines = [
        "--- Spatial strain extraction + relaxation summary ---",
        f"Interface:              {interface.name}",
        f"Polynomial degree:      {poly_degree}",
        f"phi0 (deg):             {phi0_deg}",
        f"|theta| range:          "
        f"[{np.nanmin(np.abs(theta)):.3f}, {np.nanmax(np.abs(theta)):.3f}]",
        f"|theta| mean:           {np.nanmean(np.abs(theta)):.3f}",
        f"eps_c mean / std (%):   "
        f"{np.nanmean(eps_c) * 100:.4f} / {np.nanstd(eps_c) * 100:.4f}",
        f"eps_s mean / std (%):   "
        f"{np.nanmean(eps_s) * 100:.4f} / {np.nanstd(eps_s) * 100:.4f}",
    ]
    if result is not None:
        summary_lines += [
            f"Mesh vertices:          {mesh.n_vertices}",
            f"Energy reduction:       {100 * result.energy_reduction:.1f}%",
            f"Relaxation wall time:   {elapsed:.1f} s",
        ]
    print("\n" + "\n".join(summary_lines))
    (OUT_DIR / f"{slug}_spatial_summary.txt").write_text(
        "\n".join(summary_lines) + "\n"
    )

    if args.no_plots:
        return

    _plot_headline(
        fringes, X, Y, strain, mesh, result,
        OUT_DIR / f"{slug}_spatial_relaxation.png",
        poly_degree=poly_degree,
        phi0_deg=phi0_deg,
    )


if __name__ == "__main__":
    main()
