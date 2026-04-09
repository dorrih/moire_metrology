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
from moire_metrology.discretization import Discretization
from moire_metrology.lattice import HexagonalLattice, MoireGeometry
from moire_metrology.mesh import generate_finite_mesh
from moire_metrology.plotting import plot_scalar_field
from moire_metrology.strain import (
    FringeSet,
    compute_strain_field,
    convex_hull_mask,
    displacement_from_strain_field,
)


# --- Parameters ---------------------------------------------------------

#: Path to the maintainer-only ``.mat`` polyline data file (gitignored).
DATA_PATH = (Path(__file__).parents[1]
             / "docs_internal" / "Bilayer--183_1_pointsList.mat")

#: Polynomial degree for the I(r), J(r) registry fits. The ACS Nano
#: paper Methods section uses n=11; we match it.
POLYNOMIAL_DEGREE = 11

#: Common substrate orientation φ₀. The paper Fig 1 caption reports
#: -65.6° (extracted by the maintainer's MATLAB pipeline).
PHI0_DEG_DEFAULT = -65.6

#: Layer lattice constants in the (alpha1 < alpha2) convention used by
#: :func:`get_strain`. For H-MoSe2/WSe2 the larger constant is MoSe2.
ALPHA1 = WSE2.lattice_constant   # 0.3282 nm
ALPHA2 = MOSE2.lattice_constant  # 0.3288 nm

#: Strain map grid resolution for the headline figure.
N_GRID = 100

#: Mesh sizing for the relaxation step.
N_CELLS = 55
PIXEL_SIZE = 4.0

#: Pseudo-dynamics iteration cap. Empirically, the relaxation hits a
#: numerical-noise plateau around iter 60-70 on this dataset; running
#: more is wasted work.
MAX_ITER = 70

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------


def _build_mesh_for_data(fringes: FringeSet, theta_avg_deg: float):
    """Build a finite mesh covering the data extent at the average geometry."""
    delta_avg = ALPHA2 / ALPHA1 - 1.0
    lattice = HexagonalLattice(alpha=ALPHA1)
    geom = MoireGeometry(lattice, theta_twist=theta_avg_deg, delta=delta_avg)
    mesh = generate_finite_mesh(geom, n_cells=N_CELLS, pixel_size=PIXEL_SIZE)

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
        f"polynomial degree {POLYNOMIAL_DEGREE}, φ₀ = {PHI0_DEG_DEFAULT:.1f}°"
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
        "Spatially-varying strain extraction + relaxation — H-MoSe2/WSe2\n"
        "Top row + bottom-left reproduce Halbertal et al. ACS Nano 16, 1471 "
        "(2022) Fig 1c-e. Bottom-right is the new relaxation prediction.",
        fontsize=11, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plotting (useful for headless smoke testing).",
    )
    parser.add_argument(
        "--no-relax", action="store_true",
        help="Skip the relaxation step (only do strain extraction).",
    )
    args = parser.parse_args()

    # ----- Step 1: load polylines into a FringeSet -----
    if not DATA_PATH.exists():
        print(
            f"Data file {DATA_PATH} not found.\n"
            "This example requires the maintainer-only .mat file. "
            "See the module docstring for the expected format."
        )
        return
    print(f"Loading polylines from {DATA_PATH}")
    fringes = FringeSet.from_matlab(DATA_PATH)
    print(f"  family 1 (I): {len(fringes.i_fringes)} polylines")
    print(f"  family 2 (J): {len(fringes.j_fringes)} polylines")

    # ----- Step 2: fit registry polynomials I(r), J(r) -----
    print(f"\nFitting registry polynomials of degree {POLYNOMIAL_DEGREE}...")
    I_field, J_field = fringes.fit_registry_fields(order=POLYNOMIAL_DEGREE)

    # ----- Step 3: compute strain field on a regular grid -----
    print("Computing strain field on a regular grid...")
    xs = np.concatenate([f.x for f in fringes.fringes])
    ys = np.concatenate([f.y for f in fringes.fringes])
    x_grid = np.linspace(xs.min(), xs.max(), N_GRID)
    y_grid = np.linspace(ys.min(), ys.max(), N_GRID)
    X, Y = np.meshgrid(x_grid, y_grid)

    inside_grid = convex_hull_mask(xs, ys, X, Y)

    strain = compute_strain_field(
        X, Y, I_field, J_field,
        alpha1=ALPHA1, alpha2=ALPHA2, phi0_deg=PHI0_DEG_DEFAULT,
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
        # ----- Step 4: build mesh and FEM gradient operators -----
        theta_avg = float(np.nanmean(np.abs(theta)))
        print(f"\nBuilding finite mesh at θ_avg = {theta_avg:.3f}°...")
        mesh, geom = _build_mesh_for_data(fringes, theta_avg)
        disc = Discretization(mesh, geom)
        print(f"  Mesh: {mesh.n_vertices} verts, {mesh.n_triangles} triangles")
        print(f"  Average geometry: λ = {geom.wavelength:.2f} nm")

        # ----- Step 5: build IC by evaluating strain at mesh vertices,
        # masking to the data hull, and integrating the gradient field. -----
        print("Evaluating local strain field at mesh vertices...")
        vertex_strain = compute_strain_field(
            mesh.points[0], mesh.points[1], I_field, J_field,
            alpha1=ALPHA1, alpha2=ALPHA2, phi0_deg=PHI0_DEG_DEFAULT,
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

        print("Building IC via gradient integration of (δθ, S)...")
        # Pin the vertex closest to the data centroid to fix the global
        # translation gauge.
        cx, cy = float(xs.mean()), float(ys.mean())
        d2 = (mesh.points[0] - cx) ** 2 + (mesh.points[1] - cy) ** 2
        pin_vertex = int(np.argmin(d2))

        ux, uy = displacement_from_strain_field(
            disc,
            theta_deg=theta_v, theta_avg_deg=theta_avg,
            S11=S11_v, S12=S12_v, S22=S22_v,
            pin_vertex=pin_vertex,
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

        # ----- Step 6: relaxation via pseudo_dynamics -----
        cfg = SolverConfig(
            method="pseudo_dynamics",
            pixel_size=PIXEL_SIZE,
            max_iter=MAX_ITER,
            gtol=1e-4,
            display=True,
        )
        print(f"\nRunning relaxation (max {MAX_ITER} pseudo_dynamics steps)...")
        t0 = perf_counter()
        result = RelaxationSolver(cfg).solve(
            moire_interface=MOSE2_WSE2_H_INTERFACE,
            theta_twist=geom.theta_twist,
            mesh=mesh,
            initial_solution=U_full,
        )
        elapsed = perf_counter() - t0
        print(
            f"  Done in {elapsed:.1f}s, "
            f"energy reduction = {100 * result.energy_reduction:.1f}%"
        )

    # ----- Scalar summary -----
    summary_lines = [
        "--- Spatial strain extraction + relaxation summary ---",
        f"Polynomial degree:      {POLYNOMIAL_DEGREE}",
        f"phi0 (deg):             {PHI0_DEG_DEFAULT}",
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
    (OUT_DIR / "spatial_strain_summary.txt").write_text(
        "\n".join(summary_lines) + "\n"
    )

    if args.no_plots:
        return

    _plot_headline(
        fringes, X, Y, strain, mesh, result,
        OUT_DIR / "spatial_strain_relaxation.png",
    )


if __name__ == "__main__":
    main()
