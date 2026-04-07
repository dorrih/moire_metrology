"""Strain extraction and spatially-varying constrained relaxation.

A three-part example. Part A is the inverse strain-extraction
calculation; Parts B and C are the constrained finite-mesh
relaxation workflow that the package's `PinningMap` and
`generate_finite_mesh` enable.

Part A — Strain extraction sweep (no relaxation)
================================================
Reproduces (qualitatively) Figure 3c-d of:

    Halbertal, Shabani, Pasupathy, Basov, "Extracting the Strain Matrix
    and Twist Angle from the Moiré Superlattice in van der Waals
    Heterostructures", ACS Nano 16, 1471 (2022).
    https://doi.org/10.1021/acsnano.1c09789

Given a measured moiré superlattice characterized by the two moiré
lattice vectors (lengths λ₁, λ₂ and angles φ₁, φ₂) plus the atomic
lattice constants α₁, α₂, the inverse problem is to recover the twist
angle θ and the heterostrain tensor of the layers. Sweeps Δφ ≡ φ₂ − φ₁
at fixed λ₁ = λ₂ and plots recovered θ, ε_c, ε_s. At Δφ = 60° the
unstrained-symmetric solution is recovered exactly.

Part B — Imposed uniform heterostrain via pinned AA points
==========================================================
A finite-mesh constrained relaxation that demonstrates how non-trivial
elongated domain-wall structures emerge when the pinned stacking
positions encode a uniform heterostrain.

Setup: TBG (twisted bilayer graphene) at the magic-angle-adjacent twist
θ = 1°, on a finite mesh covering several moiré cells. We enumerate
the natural AA stacking sites of the unstrained moiré, then PIN those
vertices to displacement targets `u(r) = ε · (r − r_c)` corresponding
to a 1% uniaxial heterostrain about the cell centre. The relaxation
finds the displacement field that satisfies these constraints while
minimising elastic + GSFE energy. The result is the classic AB/BA
triangular domain network — but elongated in the strain direction,
with sharp domain walls along the principal strain axis.

Part C — Imposed twist gradient via pinned AA points
====================================================
Same mechanism as Part B, but instead of a uniform strain we impose a
spatially-varying effective twist. The pinned displacement targets
`u(r) = δθ(x) · J · (r − r_c)` (where `J = [[0,−1],[1,0]]` is the 2D
rotation generator) correspond to a local twist that varies linearly
from 0.2° at one edge of the cell to 1.3° at the other, around an
average of 0.75°. The local moiré wavelength accordingly varies from
~70 nm to ~11 nm across the cell. The relaxation produces a
non-uniform AB/BA pattern with compressed/expanded moiré on each side
of the centre — exactly the kind of pattern seen in experimental
images that contain twist-angle disorder.

This is the spirit of the constrained-relaxation workflow used in
Shabani, Halbertal et al., *Deep moiré potentials in twisted
transition metal dichalcogenide bilayers*, Nature Physics 17, 720–725
(2021), where AA / XX' stacking sites identified in an STM topograph
are pinned and the rest of the displacement field is computed by
relaxation.

Outputs (saved to examples/output/):
    strain_extraction_sweep.png            — recovered θ, ε_c, ε_s vs Δφ
    strain_pinned_heterostrain.png         — Part B: uniform heterostrain
    strain_pinned_twist_gradient.png       — Part C: linear twist gradient
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from moire_metrology import GRAPHENE, RelaxationSolver, SolverConfig
from moire_metrology.discretization import Discretization, PinnedConstraints
from moire_metrology.lattice import HexagonalLattice, MoireGeometry
from moire_metrology.mesh import generate_finite_mesh
from moire_metrology.strain import get_strain_minimize_compression
from moire_metrology.strain.extraction import shear_strain_invariant


OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# --- Part A parameters --------------------------------------------------
# Match the paper's Fig. 3 setup: graphene on graphene (no lattice
# mismatch), λ₁ = λ₂ = 10 nm, φ₁ = 0°.
ALPHA = GRAPHENE.lattice_constant   # 0.247 nm
LAMBDA1 = LAMBDA2 = 10.0            # nm
PHI1_DEG = 0.0
DPHI_GRID = np.linspace(30.0, 90.0, 121)   # sweep Δφ around the unstrained 60°

# --- Part B parameters: uniform heterostrain on TBG ---------------------
# θ=1° puts us close to the magic-angle regime; the moire wavelength is
# ~14 nm, so a 5x5-cell domain is ~70 nm.
THETA_B = 1.0            # degrees
N_CELLS_B = 5
PIXEL_SIZE_B = 1.0       # nm
HETEROSTRAIN_EPS_B = 0.01     # 1% uniaxial heterostrain along x

# --- Part C parameters: linear twist gradient ---------------------------
# Local twist varies linearly from THETA_C_MIN at one edge to THETA_C_MAX
# at the other; the geometry uses the average as its reference twist.
THETA_C_MIN = 0.2        # degrees, at x = x_min
THETA_C_MAX = 1.3        # degrees, at x = x_max
THETA_C_AVG = 0.5 * (THETA_C_MIN + THETA_C_MAX)   # 0.75°
# Domain needs to be wide enough to host one moire wavelength at the
# 0.2° end (~70 nm); 4 cells of the average wavelength (~19 nm × 4 ≈ 75)
# is a sensible compromise.
N_CELLS_C = 4
PIXEL_SIZE_C = 2.0       # nm — coarse so the relaxation is fast


# =======================================================================
# Part A — Strain extraction sweep
# =======================================================================

def part_a_strain_sweep() -> None:
    """Sweep Δφ and plot recovered strain quantities, ACS Nano Fig. 3c-d."""
    print("=== Part A: strain extraction sweep ===")

    theta_recovered = np.zeros_like(DPHI_GRID)
    eps_c = np.zeros_like(DPHI_GRID)
    eps_s = np.zeros_like(DPHI_GRID)
    phi0 = np.zeros_like(DPHI_GRID)

    for k, dphi in enumerate(DPHI_GRID):
        result = get_strain_minimize_compression(
            alpha1=ALPHA, alpha2=ALPHA,
            lambda1=LAMBDA1, lambda2=LAMBDA2,
            phi1_deg=PHI1_DEG, phi2_deg=PHI1_DEG + dphi,
        )
        theta_recovered[k] = result.theta_twist
        eps_c[k] = result.eps_c
        # NOTE: we use the closed-form Eq 6 helper for ε_s rather than
        # result.eps_s. The latter is computed from the principal strains
        # of the recovered tensor, but currently has a φ₀-dependence bug
        # that the closed-form invariant does not have. See backlog
        # (get_strain_minimize_compression eps_s bug).
        eps_s[k] = shear_strain_invariant(
            alpha1=ALPHA, alpha2=ALPHA,
            lambda1=LAMBDA1, lambda2=LAMBDA2,
            phi1_deg=PHI1_DEG, phi2_deg=PHI1_DEG + dphi,
        )
        phi0[k] = result.phi0

    # Analytic prediction at Δφ = 60° (the unstrained-symmetric case):
    #   λ = α / (2 sin(θ/2))   ⇒   θ = 2 arcsin(α / (2λ))
    theta_unstrained_deg = 2.0 * np.degrees(np.arcsin(ALPHA / (2.0 * LAMBDA1)))

    # Sanity check: at Δφ = 60° the recovered twist should match analytics
    # and both strain components should vanish (the unstrained-symmetric
    # configuration).
    idx_60 = int(np.argmin(np.abs(DPHI_GRID - 60.0)))
    print(f"  λ₁ = λ₂ = {LAMBDA1} nm,  α = {ALPHA} nm")
    print(f"  Analytic θ at Δφ=60°:        {theta_unstrained_deg:.6f}°")
    print(f"  Recovered θ at Δφ=60°:       {theta_recovered[idx_60]:.6f}°")
    print(f"  ε_c (compression) at 60°:    {abs(eps_c[idx_60]):.2e}")
    print(f"  ε_s (shear, Eq 6) at 60°:    {abs(eps_s[idx_60]):.2e}")

    # ----- Plots ------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # (c) — recovered twist angle vs Δφ
    ax = axes[0]
    ax.plot(DPHI_GRID, theta_recovered, "b-", lw=1.5, label="recovered θ")
    ax.axhline(theta_unstrained_deg, color="k", ls="--", lw=0.8,
               label=f"λ = α/(2 sin θ/2)  ⇒  θ = {theta_unstrained_deg:.3f}°")
    ax.axvline(60.0, color="grey", ls=":", lw=0.7)
    ax.set_xlabel("Δφ = φ₂ − φ₁  (deg)")
    ax.set_ylabel("recovered twist angle θ  (deg)")
    ax.set_title("Strain extraction: recovered θ vs Δφ\n(ACS Nano Fig. 3c)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # (d) — recovered ε_c, ε_s vs Δφ
    ax = axes[1]
    ax.plot(DPHI_GRID, eps_c * 100, "k-", lw=1.5, label="ε_c (compression)")
    ax.plot(DPHI_GRID, eps_s * 100, "r-", lw=1.5, label="ε_s (shear)")
    ax.axvline(60.0, color="grey", ls=":", lw=0.7)
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("Δφ = φ₂ − φ₁  (deg)")
    ax.set_ylabel("strain (%)")
    ax.set_title(
        "Recovered strain vs Δφ  (ACS Nano Fig. 3d)\n"
        "ε_c minimized by choice of φ₀; ε_s is φ₀-invariant"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = OUT_DIR / "strain_extraction_sweep.png"
    fig.savefig(out_path, dpi=150)
    print(f"  Saved {out_path}")


# =======================================================================
# Pinning helpers (shared by Parts B and C)
# =======================================================================

def _natural_aa_positions(geometry: MoireGeometry, mesh,
                          extent: int = 12) -> np.ndarray:
    """Enumerate natural AA stacking positions of the moiré that lie
    inside the bounding box of the supplied mesh.

    The natural AA points of an unstrained moiré sit at integer linear
    combinations of the moiré lattice vectors V1, V2 (which the geometry
    object exposes as ``geometry.V1``, ``geometry.V2``). We sweep
    (i, j) ∈ [-extent, extent]² and keep the points whose physical
    position falls inside the mesh.

    Returns an array of shape (N, 2): N AA positions, each (x, y).
    """
    V1m = geometry.V1
    V2m = geometry.V2
    x_min, x_max = mesh.points[0].min(), mesh.points[0].max()
    y_min, y_max = mesh.points[1].min(), mesh.points[1].max()

    aa = []
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            x = i * V1m[0] + j * V2m[0]
            y = i * V1m[1] + j * V2m[1]
            if x_min <= x <= x_max and y_min <= y <= y_max:
                aa.append((x, y))
    return np.asarray(aa)


def _nearest_vertex_indices(mesh, positions: np.ndarray) -> np.ndarray:
    """For each (x, y) in positions, return the index of the closest
    mesh vertex. Used to map AA-position targets onto actual DOFs."""
    out = np.empty(len(positions), dtype=int)
    for k, (xs, ys) in enumerate(positions):
        d = (mesh.points[0] - xs) ** 2 + (mesh.points[1] - ys) ** 2
        out[k] = int(np.argmin(d))
    return out


def _build_displacement_pins(conv, vertex_indices: np.ndarray,
                             ux_targets: np.ndarray,
                             uy_targets: np.ndarray) -> PinnedConstraints:
    """Build a PinnedConstraints object that pins specific vertices to
    specific *relative* displacements (ux, uy).

    For a 1+1 bilayer stack the relative displacement u between the
    two layers is split symmetrically: layer 1 gets +u/2, layer 2 gets
    −u/2. This matches the convention in PinningMap._pin_vertex_dofs.
    The result can be passed to RelaxationSolver.solve(constraints=...).

    This is a thin lower-level alternative to PinningMap.pin_stacking()
    for the case where the pinned target is a known displacement value
    (e.g. derived from an externally-imposed strain or twist gradient)
    rather than a stacking label.
    """
    Nv = conv.n_vertices
    nlayer1 = conv.nlayer1
    nlayer2 = conv.nlayer2
    nlayers_total = nlayer1 + nlayer2
    n_full = conv.n_sol

    inner1 = nlayer1 - 1
    inner2 = nlayer1

    pinned_dofs: dict[int, float] = {}
    for vi, ux, uy in zip(vertex_indices, ux_targets, uy_targets):
        # Layer 1 inner: ux1 = +ux/2, uy1 = +uy/2
        pinned_dofs[inner1 * Nv + int(vi)] = +ux / 2.0
        pinned_dofs[nlayers_total * Nv + inner1 * Nv + int(vi)] = +uy / 2.0
        # Layer 2 inner: ux2 = -ux/2, uy2 = -uy/2
        pinned_dofs[inner2 * Nv + int(vi)] = -ux / 2.0
        pinned_dofs[nlayers_total * Nv + inner2 * Nv + int(vi)] = -uy / 2.0

    pinned_indices = np.array(sorted(pinned_dofs.keys()), dtype=int)
    pinned_values = np.array([pinned_dofs[i] for i in pinned_indices])
    free_indices = np.setdiff1d(np.arange(n_full), pinned_indices)

    return PinnedConstraints(
        free_indices=free_indices,
        pinned_indices=pinned_indices,
        pinned_values=pinned_values,
        n_free=len(free_indices),
        n_full=n_full,
    )


def _plot_pinned_relaxation(mesh, geometry, result, pinned_vertex_indices,
                            *, title: str, out_path: Path) -> None:
    """Side-by-side unrelaxed vs relaxed stacking-energy map with the
    pinned vertices marked. Shared by Parts B and C."""
    from moire_metrology.gsfe import GSFESurface

    x = mesh.points[0]
    y = mesh.points[1]
    gsfe = GSFESurface(GRAPHENE.gsfe_coeffs)
    v0, w0 = geometry.stacking_phases(x, y)
    g_unrelaxed = gsfe(v0, w0)
    g_relaxed = result.gsfe_map
    vmax = float(np.percentile(g_unrelaxed, 95))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, g, label in [
        (axes[0], g_unrelaxed, "Unrelaxed (rigid stacking)"),
        (axes[1], g_relaxed,
         f"Relaxed ({100 * result.energy_reduction:.0f}% energy reduction)"),
    ]:
        sc = ax.scatter(x, y, c=g, cmap="magma", s=14, vmin=0.0, vmax=vmax)
        ax.scatter(
            x[pinned_vertex_indices], y[pinned_vertex_indices],
            marker="o", s=70, edgecolor="cyan", facecolor="none",
            linewidth=1.2, label="pinned vertices",
        )
        ax.set_aspect("equal")
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.set_title(label, fontsize=11)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        plt.colorbar(sc, ax=ax, label="GSFE (meV/nm²)")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Saved {out_path}")


# =======================================================================
# Part B — Imposed uniform heterostrain via pinned AA points
# =======================================================================

def part_b_uniform_heterostrain() -> None:
    """Pin natural AA points to a uniform-heterostrain displacement field.

    The displacement target at position r is

        u(r) = ε · (r − r_c)

    with ε a small uniaxial heterostrain tensor (1% along x here) and
    r_c the centre of the mesh. The relaxation must find a smooth
    displacement field that satisfies these constraints at the pinned
    vertices and minimises elastic + GSFE energy elsewhere; the result
    shows AB/BA domains elongated along the y axis (the unstrained
    direction).
    """
    print("\n=== Part B: imposed uniform heterostrain ===")

    lattice = HexagonalLattice(alpha=GRAPHENE.lattice_constant)
    geometry = MoireGeometry(lattice, theta_twist=THETA_B)
    mesh = generate_finite_mesh(
        geometry, n_cells=N_CELLS_B, pixel_size=PIXEL_SIZE_B,
    )
    print(f"  TBG @ θ = {THETA_B}°,  λ_moire = {geometry.wavelength:.2f} nm")
    print(f"  Mesh: {mesh.n_vertices} vertices, "
          f"{mesh.n_triangles} triangles  ({N_CELLS_B}x{N_CELLS_B} moire cells)")

    # Natural AA positions and the centre of the mesh
    aa_positions = _natural_aa_positions(geometry, mesh)
    r_c = np.array([mesh.points[0].mean(), mesh.points[1].mean()])
    print(f"  Natural AA points inside mesh: {len(aa_positions)}")

    # Uniaxial heterostrain ε along x
    eps = np.array([[HETEROSTRAIN_EPS_B, 0.0],
                    [0.0,                0.0]])

    dr = aa_positions - r_c                # (N, 2)
    u_targets = (eps @ dr.T).T             # (N, 2)
    ux_targets = u_targets[:, 0]
    uy_targets = u_targets[:, 1]
    print(f"  Heterostrain: {100 * HETEROSTRAIN_EPS_B:.1f}% uniaxial along x")
    print(f"  Max |u_target| at pinned points: "
          f"{float(np.linalg.norm(u_targets, axis=1).max()) * 1000:.2f} pm")

    vertex_indices = _nearest_vertex_indices(mesh, aa_positions)

    disc = Discretization(mesh, geometry)
    conv = disc.build_conversion_matrices(nlayer1=1, nlayer2=1)
    constraints = _build_displacement_pins(
        conv, vertex_indices, ux_targets, uy_targets,
    )
    print(f"  Pinned DOFs: {len(constraints.pinned_indices)} of {constraints.n_full}")

    cfg = SolverConfig(
        method="L-BFGS-B", pixel_size=PIXEL_SIZE_B, max_iter=400,
        gtol=1e-5, display=False,
    )
    print("  Running L-BFGS-B relaxation...")
    t0 = perf_counter()
    result = RelaxationSolver(cfg).solve(
        material1=GRAPHENE, material2=GRAPHENE, theta_twist=THETA_B,
        mesh=mesh, constraints=constraints,
    )
    print(f"  Done in {perf_counter() - t0:.2f} s")
    print(f"  Energy reduction: {100 * result.energy_reduction:.1f}%")

    _plot_pinned_relaxation(
        mesh, geometry, result, vertex_indices,
        title=(f"Pinned TBG relaxation with {100 * HETEROSTRAIN_EPS_B:.1f}% "
               f"uniaxial heterostrain (θ = {THETA_B}°)\n"
               f"AA points pinned to u(r) = ε · (r − r_c)"),
        out_path=OUT_DIR / "strain_pinned_heterostrain.png",
    )


# =======================================================================
# Part C — Imposed twist gradient via pinned AA points
# =======================================================================

def part_c_twist_gradient() -> None:
    """Pin natural AA points to a displacement field consistent with
    a linearly-varying local twist angle.

    The geometry uses the AVERAGE twist as its reference. The pinned
    displacement target at position r is

        u(r) = δθ(x) · J · (r − r_c)

    where J = [[0, −1], [1, 0]] is the 2D rotation generator and
    δθ(x) is the local deviation of the twist from the average:

        δθ(x) = (θ_max − θ_min) · (x − x_c) / W      (in radians)

    Equivalent to "rotate layer 2 about r_c by an extra δθ(x), at each
    pinned site". The relaxation produces a non-uniform AB/BA pattern
    with the moiré pattern compressed where δθ is positive (more twist,
    smaller wavelength) and stretched where δθ is negative.
    """
    print("\n=== Part C: imposed linear twist gradient ===")

    lattice = HexagonalLattice(alpha=GRAPHENE.lattice_constant)
    geometry = MoireGeometry(lattice, theta_twist=THETA_C_AVG)
    mesh = generate_finite_mesh(
        geometry, n_cells=N_CELLS_C, pixel_size=PIXEL_SIZE_C,
    )
    print(f"  Reference (average) twist: θ_avg = {THETA_C_AVG}°")
    print(f"  Local twist range: {THETA_C_MIN}° → {THETA_C_MAX}°")
    print(f"  Average moire wavelength: {geometry.wavelength:.2f} nm")
    print(f"  Mesh: {mesh.n_vertices} vertices, "
          f"{mesh.n_triangles} triangles")

    aa_positions = _natural_aa_positions(geometry, mesh)
    r_c = np.array([mesh.points[0].mean(), mesh.points[1].mean()])
    print(f"  Natural AA points inside mesh: {len(aa_positions)}")

    # Linear twist gradient along x: δθ(x) varies from -δθ_max at x_min
    # to +δθ_max at x_max, around δθ=0 at the centre.
    x_min = mesh.points[0].min()
    x_max = mesh.points[0].max()
    W = x_max - x_min
    dtheta_max_deg = 0.5 * (THETA_C_MAX - THETA_C_MIN)
    dtheta_max_rad = np.radians(dtheta_max_deg)

    dx = aa_positions[:, 0] - r_c[0]
    dy = aa_positions[:, 1] - r_c[1]
    # δθ at each AA point, in radians
    dtheta = (2.0 * dtheta_max_rad / W) * dx
    # u = δθ · J · (r − r_c)  with J = [[0, -1], [1, 0]]
    ux_targets = -dtheta * dy
    uy_targets = +dtheta * dx
    u_norm_max = float(np.sqrt(ux_targets ** 2 + uy_targets ** 2).max())
    print(f"  Max |u_target| at pinned points: {u_norm_max * 1000:.2f} pm")

    vertex_indices = _nearest_vertex_indices(mesh, aa_positions)

    disc = Discretization(mesh, geometry)
    conv = disc.build_conversion_matrices(nlayer1=1, nlayer2=1)
    constraints = _build_displacement_pins(
        conv, vertex_indices, ux_targets, uy_targets,
    )
    print(f"  Pinned DOFs: {len(constraints.pinned_indices)} of {constraints.n_full}")

    cfg = SolverConfig(
        method="L-BFGS-B", pixel_size=PIXEL_SIZE_C, max_iter=600,
        gtol=1e-5, display=False,
    )
    print("  Running L-BFGS-B relaxation...")
    t0 = perf_counter()
    result = RelaxationSolver(cfg).solve(
        material1=GRAPHENE, material2=GRAPHENE, theta_twist=THETA_C_AVG,
        mesh=mesh, constraints=constraints,
    )
    print(f"  Done in {perf_counter() - t0:.2f} s")
    print(f"  Energy reduction: {100 * result.energy_reduction:.1f}%")

    _plot_pinned_relaxation(
        mesh, geometry, result, vertex_indices,
        title=(f"Pinned TBG relaxation with linear twist gradient "
               f"({THETA_C_MIN}° → {THETA_C_MAX}° across the cell)\n"
               f"AA points pinned to u(r) = δθ(x) · J · (r − r_c)"),
        out_path=OUT_DIR / "strain_pinned_twist_gradient.png",
    )


# =======================================================================

def main() -> None:
    part_a_strain_sweep()
    part_b_uniform_heterostrain()
    part_c_twist_gradient()


if __name__ == "__main__":
    main()
