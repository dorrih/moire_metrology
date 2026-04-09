"""Spatially-varying strain extraction from traced moire fringes.

Reproduces Fig. 1c-e of Halbertal, Shabani, Pasupathy & Basov, ACS Nano
16, 1471 (2022) on a real H-stacked MoSe2/WSe2 sample with non-uniform
twist angle and strain across the field of view. The recovered θ(x, y),
ε_c(x, y), and ε_s(x, y) maps come out of the package's
``moire_metrology.strain`` API in three calls.

Pipeline:

1. :class:`FringeSet` loads two families of integer-labeled polylines
   from a MATLAB ``.mat`` file (or any other source).
2. :meth:`FringeSet.fit_registry_fields` fits the two registry index
   polynomials ``I(x, y)`` and ``J(x, y)`` to all polyline points.
3. :func:`compute_strain_field` recovers ``(θ, ε_c, ε_s)(x, y)`` from
   ``(I, J)`` via the eq. 9 inversion of the paper.
4. :func:`convex_hull_mask` confines the strain map to the convex hull
   of the input polylines — a high-degree polynomial extrapolates
   wildly outside its data support.

Headline figure (saved to ``examples/output/``): a 1×3 panel with
``θ(x, y)``, ``ε_c(x, y)``, ``ε_s(x, y)``, matching the paper colour
limits in Fig. 1c-e.

Extending this to a relaxation prediction (the natural next step — use
the recovered registry field to drive a constrained relaxation against
``MOSE2_WSE2_H_INTERFACE``) is a separate research thread; the current
``examples/spatial_strain_relaxation.py`` is the API template, the
``compute_displacement_field`` and pinning machinery are in place, but
the constant-θ relaxation framework needs more work to digest a
spatially varying twist field cleanly.

References
----------
* Halbertal, Shabani, Pasupathy & Basov, "Extracting the strain matrix
  and twist angle from the moiré superlattice in van der Waals
  heterostructures", ACS Nano 16, 1471 (2022),
  doi:10.1021/acsnano.1c09789.

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

import matplotlib.pyplot as plt
import numpy as np

from moire_metrology import MOSE2, WSE2
from moire_metrology.strain import (
    FringeSet,
    compute_strain_field,
    convex_hull_mask,
)


# --- Parameters ---------------------------------------------------------

#: Path to the maintainer-only ``.mat`` polyline data file (gitignored).
DATA_PATH = (Path(__file__).parents[1]
             / "docs_internal" / "Bilayer--183_1_pointsList.mat")

#: Polynomial degree for the I(r), J(r) registry fits. The ACS Nano
#: paper Methods section uses n=11; we match it.
POLYNOMIAL_DEGREE = 11

#: Common substrate orientation φ₀. The paper Fig 1 caption reports
#: -65.6° (extracted by the maintainer's MATLAB pipeline as
#: ``phi0_preset = 5.138327499706437`` rad).
PHI0_DEG_DEFAULT = -65.6

#: Layer lattice constants in the (alpha1 < alpha2) convention used by
#: :func:`get_strain`. For H-MoSe2/WSe2 the larger constant is MoSe2.
ALPHA1 = WSE2.lattice_constant   # 0.3282 nm
ALPHA2 = MOSE2.lattice_constant  # 0.3288 nm

#: Strain map grid resolution.
N_GRID = 100

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_headline(
    fringes: FringeSet,
    X: np.ndarray, Y: np.ndarray,
    strain: dict,
    out_path: Path,
) -> None:
    """1x3 panel reproducing paper Fig. 1c-e on the input polyline data."""
    xs_I = np.concatenate([f.x for f in fringes.i_fringes])
    ys_I = np.concatenate([f.y for f in fringes.i_fringes])
    xs_J = np.concatenate([f.x for f in fringes.j_fringes])
    ys_J = np.concatenate([f.y for f in fringes.j_fringes])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    extent = [X.min(), X.max(), Y.min(), Y.max()]

    # ----- recovered θ with sampled-points overlay -----
    ax = axes[0]
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

    # ----- recovered ε_c -----
    ax = axes[1]
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

    # ----- recovered ε_s -----
    ax = axes[2]
    sc = ax.imshow(
        strain["eps_s"] * 100,
        extent=extent, origin="lower",
        cmap="inferno", vmin=0, vmax=0.8,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
    ax.set_title(
        "recovered ε_s(x, y) — paper Fig 1e\n"
        f"mean = {np.nanmean(strain['eps_s']) * 100:.4f}%, "
        f"std = {np.nanstd(strain['eps_s']) * 100:.4f}%"
    )
    plt.colorbar(sc, ax=ax, label="ε_s (%)", shrink=0.8)

    fig.suptitle(
        "Spatially-varying strain extraction — H-MoSe2/WSe2\n"
        "Reproduces Halbertal et al. ACS Nano 16, 1471 (2022) Fig 1c-e",
        fontsize=12, y=1.02,
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

    # Confine the strain map to the convex hull of the polyline data —
    # the polynomial extrapolates wildly outside its data support.
    inside = convex_hull_mask(xs, ys, X, Y)

    strain = compute_strain_field(
        X, Y, I_field, J_field,
        alpha1=ALPHA1, alpha2=ALPHA2, phi0_deg=PHI0_DEG_DEFAULT,
    )
    for key in ("theta", "eps_c", "eps_s"):
        strain[key] = np.where(inside, strain[key], np.nan)

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

    # ----- Scalar summary -----
    summary_lines = [
        "--- Spatial strain extraction summary ---",
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
    print("\n" + "\n".join(summary_lines))
    (OUT_DIR / "spatial_strain_summary.txt").write_text(
        "\n".join(summary_lines) + "\n"
    )

    if args.no_plots:
        return

    _plot_headline(
        fringes, X, Y, strain,
        OUT_DIR / "spatial_strain_extraction.png",
    )


if __name__ == "__main__":
    main()
