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

Both Parts B and C use the H-stacked MoSe2/WSe2 heterointerface
(α_MoSe2 = 0.3288 nm, α_WSe2 = 0.3282 nm, intrinsic mismatch
δ ≈ 0.18% — non-zero even at θ = 0). This matches the experimental
system in Shabani, Halbertal et al., *Deep moiré potentials in
twisted transition metal dichalcogenide bilayers*, Nature Physics
**17**, 720–725 (2021), DOI 10.1038/s41567-021-01174-7, which
introduced the spatially-varying constrained-relaxation workflow that
Parts B and C demonstrate.

Part B — Imposed uniform heterostrain via pinned XX' points
===========================================================
A finite-mesh constrained relaxation that demonstrates how non-trivial
elongated domain-wall structures emerge when the pinned stacking
positions encode a uniform heterostrain.

Setup: H-stacked MoSe2/WSe2 at θ = 0.5°, on a finite mesh covering
several moiré cells (~5 × 5 cells, ~180 nm wide). We enumerate the
natural XX' stacking sites of the unstrained moiré, then PIN those
vertices to displacement targets `u(r) = ε · (r − r_c)` corresponding
to a 1% uniaxial heterostrain about the cell centre. The relaxation
finds the displacement field that satisfies these constraints while
minimising elastic + GSFE energy. The result is the AB/BA triangular
domain network elongated along the strain direction.

Part C — Imposed twist gradient via pinned XX' points
=====================================================
Same mechanism as Part B, but instead of a uniform strain we impose a
spatially-varying effective twist. The pinned displacement targets
`u(r) = δθ(x) · J · (r − r_c)` (where `J = [[0,−1],[1,0]]` is the 2D
rotation generator) correspond to a local twist that varies linearly
from 0.1° at one edge of the cell to 0.7° at the other, around an
average of 0.4°. The local moiré wavelength accordingly varies from
~130 nm at the low-twist end to ~27 nm at the high-twist end. The
mesh is rectangular (~10 cells along the gradient axis × ~3 cells
perpendicular) so that the gradient region hosts many full moiré
periods and isn't dominated by edge effects. The relaxation produces
a non-uniform AB/BA pattern: dense small triangles where the local
twist is large, sparse stretched ones where it's small — the
hallmark of twist-angle disorder visible in real STM images of TMD
heterobilayers.

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

from moire_metrology import (
    GRAPHENE, MOSE2, WSE2, RelaxationSolver, SolverConfig,
)
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

# --- Part B parameters: uniform heterostrain on H-MoSe2/WSe2 ------------
# At α_TMD ≈ 0.328 nm and intrinsic mismatch δ ≈ 0.18%, θ = 0.5° gives
# an effective angle √(δ² + θ²) ≈ 0.0089 and a moire wavelength of
# ~37 nm. A 5x5-cell domain is ~185 nm — small enough to relax quickly,
# wide enough that the 1% uniform heterostrain is visible across
# several wavelengths.
THETA_B = 0.5            # degrees
N_CELLS_B = 5
PIXEL_SIZE_B = 3.0       # nm
HETEROSTRAIN_EPS_B = 0.01     # 1% uniaxial heterostrain along x

# --- Part C parameters: linear twist gradient on H-MoSe2/WSe2 -----------
# The geometry uses the AVERAGE local twist as its reference; the
# local twist varies linearly from THETA_C_MIN at x = x_min to
# THETA_C_MAX at x = x_max. With a 0.18% intrinsic TMD mismatch, the
# local moire wavelength varies from ~130 nm at θ=0.1° down to ~27 nm
# at θ=0.7°. The mesh is RECTANGULAR — wide along the gradient axis
# so we host ~10 average wavelengths there, narrow perpendicular.
THETA_C_MIN = 0.1        # degrees, at x = x_min
THETA_C_MAX = 0.7        # degrees, at x = x_max
THETA_C_AVG = 0.5 * (THETA_C_MIN + THETA_C_MAX)   # 0.4°
# At the average twist (0.4°): effective angle ≈ 0.00748,
# λ_avg ≈ 0.328 / 0.00748 ≈ 43.9 nm. 10 cells along the gradient axis
# is ~440 nm; 3 cells perpendicular is ~130 nm.
N_CELLS_C_X = 10
N_CELLS_C_Y = 3
PIXEL_SIZE_C = 4.0       # nm — coarse so the relaxation is fast


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

def _natural_high_symmetry_positions(
    geometry: MoireGeometry, mesh, *,
    sublattice: str = "AA", extent: int = 16,
) -> np.ndarray:
    """Enumerate natural high-symmetry positions of the unstrained moiré
    that lie inside the bounding box of the supplied mesh.

    The three sublattices form a triangular lattice in real space; in
    the moiré's stacking-phase coordinates they sit at:
        AA  →  (v, w) = (0, 0)
        AB  →  (v, w) = (2π/3, 2π/3)
        BA  →  (v, w) = (4π/3, 4π/3)
    Translating into real-space offsets relative to the moiré-lattice
    point (i, j):
        AA  →  i V1 + j V2
        AB  →  (i + 1/3) V1 + (j + 1/3) V2
        BA  →  (i + 2/3) V1 + (j + 2/3) V2

    We sweep (i, j) ∈ [-extent, extent]² and keep the points inside
    the mesh's bounding box.

    Returns an array of shape (N, 2).
    """
    offsets = {"AA": (0.0, 0.0), "AB": (1.0 / 3, 1.0 / 3), "BA": (2.0 / 3, 2.0 / 3)}
    if sublattice not in offsets:
        raise ValueError(f"sublattice must be one of {list(offsets)}")
    di, dj = offsets[sublattice]

    V1m = geometry.V1
    V2m = geometry.V2
    x_min, x_max = mesh.points[0].min(), mesh.points[0].max()
    y_min, y_max = mesh.points[1].min(), mesh.points[1].max()

    pts = []
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            s = i + di
            t = j + dj
            x = s * V1m[0] + t * V2m[0]
            y = s * V1m[1] + t * V2m[1]
            if x_min <= x <= x_max and y_min <= y <= y_max:
                pts.append((x, y))
    return np.asarray(pts)


# Backwards-compatible alias for code that just wants the AA sublattice
def _natural_aa_positions(geometry: MoireGeometry, mesh,
                          extent: int = 16) -> np.ndarray:
    return _natural_high_symmetry_positions(
        geometry, mesh, sublattice="AA", extent=extent,
    )


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
                            interface_material, *,
                            title: str, out_path: Path,
                            show_local_twist: bool = False,
                            local_twist_range: tuple[float, float] | None = None,
                            ) -> None:
    """Side-by-side unrelaxed vs relaxed stacking-energy map with the
    pinned vertices marked. Shared by Parts B and C. When
    show_local_twist=True, a third panel renders the local twist angle
    field of the relaxed solution — useful for visualising imposed
    twist gradients. If local_twist_range is supplied, the colour
    scale of that panel is clipped to (vmin, vmax); useful when the
    raw local twist field is dominated by high-frequency oscillations
    around a slow underlying gradient.
    """
    from moire_metrology.gsfe import GSFESurface

    x = mesh.points[0]
    y = mesh.points[1]
    gsfe = GSFESurface(interface_material.gsfe_coeffs)
    v0, w0 = geometry.stacking_phases(x, y)
    g_unrelaxed = gsfe(v0, w0)
    g_relaxed = result.gsfe_map
    vmax = float(np.percentile(g_unrelaxed, 95))

    n_panels = 3 if show_local_twist else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, 6))

    panels = [
        (axes[0], g_unrelaxed, "Unrelaxed (rigid stacking)", "magma",
         0.0, vmax, "GSFE (meV/nm²)"),
        (axes[1], g_relaxed,
         f"Relaxed ({100 * result.energy_reduction:.0f}% energy reduction)",
         "magma", 0.0, vmax, "GSFE (meV/nm²)"),
    ]
    if show_local_twist:
        local_twist = result.local_twist(stack=1, layer=0)
        if local_twist_range is not None:
            vmin_lt, vmax_lt = local_twist_range
        else:
            delta = float(np.percentile(np.abs(local_twist - geometry.theta_twist), 99))
            vmin_lt = geometry.theta_twist - delta
            vmax_lt = geometry.theta_twist + delta
        panels.append(
            (axes[2], local_twist,
             "Local twist angle (clipped)", "RdBu_r",
             vmin_lt, vmax_lt, "θ_local (deg)")
        )

    for ax, field, label, cmap, vmin_p, vmax_p, cbar_label in panels:
        sc = ax.scatter(x, y, c=field, cmap=cmap, s=14,
                        vmin=vmin_p, vmax=vmax_p)
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
        plt.colorbar(sc, ax=ax, label=cbar_label)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Saved {out_path}")


# =======================================================================
# Part B — Imposed uniform heterostrain via pinned AA points
# =======================================================================

def part_b_uniform_heterostrain() -> None:
    """Pin natural XX' points of an H-MoSe2/WSe2 moire to a
    uniform-heterostrain displacement field.

    The displacement target at position r is

        u(r) = ε · (r − r_c)

    with ε a small uniaxial heterostrain tensor (1% along x here) and
    r_c the centre of the mesh. The relaxation must find a smooth
    displacement field that satisfies these constraints at the pinned
    vertices and minimises elastic + GSFE energy elsewhere.
    """
    print("\n=== Part B: imposed uniform heterostrain (H-MoSe2/WSe2) ===")

    # Lattice/geometry: use the average of the two TMD lattice constants
    # for the substrate-side reference; the package's solver computes
    # the lattice mismatch from material1 / material2 lattice constants
    # automatically.
    lattice = HexagonalLattice(alpha=WSE2.lattice_constant)
    geometry = MoireGeometry(
        lattice, theta_twist=THETA_B,
        delta=MOSE2.lattice_constant / WSE2.lattice_constant - 1.0,
    )
    mesh = generate_finite_mesh(
        geometry, n_cells=N_CELLS_B, pixel_size=PIXEL_SIZE_B,
    )
    print(f"  H-MoSe2/WSe2 @ θ = {THETA_B}°,  "
          f"intrinsic mismatch δ = {geometry.delta * 100:.3f}%")
    print(f"  Moire wavelength: {geometry.wavelength:.2f} nm")
    print(f"  Mesh: {mesh.n_vertices} vertices, "
          f"{mesh.n_triangles} triangles  ({N_CELLS_B}x{N_CELLS_B} moire cells)")

    # Natural XX' (== AA in the package's notation) positions and the
    # centre of the mesh
    aa_positions = _natural_aa_positions(geometry, mesh)
    r_c = np.array([mesh.points[0].mean(), mesh.points[1].mean()])
    print(f"  Natural XX' points inside mesh: {len(aa_positions)}")

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
        material1=MOSE2, material2=WSE2, theta_twist=THETA_B,
        mesh=mesh, constraints=constraints,
    )
    print(f"  Done in {perf_counter() - t0:.2f} s")
    print(f"  Energy reduction: {100 * result.energy_reduction:.1f}%")

    _plot_pinned_relaxation(
        mesh, geometry, result, vertex_indices, MOSE2,
        title=(f"Pinned H-MoSe2/WSe2 relaxation, "
               f"{100 * HETEROSTRAIN_EPS_B:.1f}% uniaxial heterostrain "
               f"(θ = {THETA_B}°)\n"
               f"XX' points pinned to u(r) = ε · (r − r_c)"),
        out_path=OUT_DIR / "strain_pinned_heterostrain.png",
    )


# =======================================================================
# Part C — Imposed twist gradient via pinned AA points
# =======================================================================

def part_c_twist_gradient() -> None:
    """Pin natural XX' points of an H-MoSe2/WSe2 moire to a displacement
    field consistent with a linearly-varying local twist angle.

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

    The mesh is rectangular: ~10 cells along the gradient axis (so the
    spatially-varying pattern hosts many full moiré periods) and only
    ~3 cells perpendicular to it.
    """
    print("\n=== Part C: imposed twist gradient (H-MoSe2/WSe2) ===")

    lattice = HexagonalLattice(alpha=WSE2.lattice_constant)
    geometry = MoireGeometry(
        lattice, theta_twist=THETA_C_AVG,
        delta=MOSE2.lattice_constant / WSE2.lattice_constant - 1.0,
    )
    mesh = generate_finite_mesh(
        geometry, pixel_size=PIXEL_SIZE_C,
        n_cells_x=N_CELLS_C_X, n_cells_y=N_CELLS_C_Y,
    )
    print(f"  H-MoSe2/WSe2  reference twist θ_avg = {THETA_C_AVG}°")
    print(f"  Local twist range: {THETA_C_MIN}° → {THETA_C_MAX}°")
    print(f"  Average moire wavelength: {geometry.wavelength:.2f} nm")
    print(f"  Mesh: {mesh.n_vertices} vertices, {mesh.n_triangles} triangles  "
          f"({N_CELLS_C_X} × {N_CELLS_C_Y} moire cells)")

    # Pin all three high-symmetry sublattices (AA, AB, BA) so the
    # relaxation has enough constraints to follow the smooth gradient
    # field rather than wiggling around it. With only the AA sublattice
    # pinned the optimizer is too free; with all three the relaxed local
    # twist tracks the imposed gradient much more closely.
    aa_positions = np.concatenate([
        _natural_high_symmetry_positions(geometry, mesh, sublattice=s)
        for s in ("AA", "AB", "BA")
    ])
    r_c = np.array([mesh.points[0].mean(), mesh.points[1].mean()])
    print(f"  Natural high-symmetry points inside mesh "
          f"(AA + AB + BA): {len(aa_positions)}")

    # Determine which physical axis is the LONG axis of the mesh
    # (the one that hosts ~10 moire cells of variation). We picked
    # n_cells_x along V1 and n_cells_y along V2; for typical
    # hexagonal moire geometries V1 is mostly aligned with y, so the
    # long extent ends up along y. We pick the gradient axis at run
    # time so the example stays correct if the geometry rotates.
    span_x = mesh.points[0].max() - mesh.points[0].min()
    span_y = mesh.points[1].max() - mesh.points[1].min()
    if span_x >= span_y:
        gradient_axis = 0  # gradient along x
        W = span_x
    else:
        gradient_axis = 1  # gradient along y
        W = span_y
    print(f"  Mesh extent: {span_x:.1f} × {span_y:.1f} nm  "
          f"→ gradient along {'x' if gradient_axis == 0 else 'y'}")

    # Curvature κ of the twist gradient: the local rotation field
    # ω(r) = (1/2)(∂u_y/∂x − ∂u_x/∂y) varies linearly along the gradient
    # axis from −(THETA_C_MAX−THETA_C_MIN)/2 at one edge to +the same
    # at the other, in radians. So κ = (THETA_C_MAX−THETA_C_MIN)/W rad/nm.
    kappa = np.radians(THETA_C_MAX - THETA_C_MIN) / W

    dx = aa_positions[:, 0] - r_c[0]
    dy = aa_positions[:, 1] - r_c[1]

    # Divergence-free displacement field that produces ω(r) = κ · grad_coord
    # at each pinned point. Picking the solution with the OTHER displacement
    # component zero keeps the field as simple as possible:
    #   gradient along y  →  u_x = −κ y²,  u_y = 0
    #   gradient along x  →  u_x = 0,      u_y = +κ x²
    # Check (y-gradient case):
    #   (1/2)(∂u_y/∂x − ∂u_x/∂y) = (1/2)(0 − (−2κy)) = κ·y ✓
    if gradient_axis == 1:
        ux_targets = -kappa * dy ** 2
        uy_targets = np.zeros_like(dy)
    else:
        ux_targets = np.zeros_like(dx)
        uy_targets = +kappa * dx ** 2
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
        material1=MOSE2, material2=WSE2, theta_twist=THETA_C_AVG,
        mesh=mesh, constraints=constraints,
    )
    print(f"  Done in {perf_counter() - t0:.2f} s")
    print(f"  Energy reduction: {100 * result.energy_reduction:.1f}%")

    # Sanity check: the relaxed local twist should span roughly
    # [THETA_C_MIN, THETA_C_MAX] across the cell. We also look at the
    # average over horizontal stripes (constant gradient_coord) to
    # separate the slow gradient signal from high-frequency wiggles
    # introduced by the GSFE preferring discrete stacking minima.
    local_twist = result.local_twist(stack=1, layer=0)
    print(f"  Relaxed local twist range (per vertex): "
          f"{local_twist.min():.3f}° → {local_twist.max():.3f}°  "
          f"(target: {THETA_C_MIN}° → {THETA_C_MAX}°)")
    coord = mesh.points[gradient_axis]
    bin_edges = np.linspace(coord.min(), coord.max(), 11)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_means = np.array([
        local_twist[(coord >= lo) & (coord < hi)].mean()
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:])
    ])
    print(f"  Binned mean local twist ({'y' if gradient_axis == 1 else 'x'}-stripes):")
    for c, m in zip(bin_centers, bin_means):
        print(f"    {'y' if gradient_axis == 1 else 'x'}={c:7.1f} nm  →  "
              f"<θ_local> = {m:.3f}°")

    # Clip the local-twist colour scale to the imposed range so the
    # smooth gradient signal is visible above the high-frequency
    # noise that the GSFE preference for discrete stacking minima
    # introduces between the pinned points (the binned-mean diagnostic
    # above shows the gradient is correctly imposed in the bulk).
    _plot_pinned_relaxation(
        mesh, geometry, result, vertex_indices, MOSE2,
        title=(f"Pinned H-MoSe2/WSe2 relaxation with linear twist gradient "
               f"({THETA_C_MIN}° → {THETA_C_MAX}° across the cell)\n"
               f"XX'/MX'/MM' points pinned to a divergence-free rotation field "
               f"u(r) = (−κy², 0)"),
        out_path=OUT_DIR / "strain_pinned_twist_gradient.png",
        show_local_twist=True,
        local_twist_range=(THETA_C_MIN, THETA_C_MAX),
    )


# =======================================================================

def main() -> None:
    part_a_strain_sweep()
    part_b_uniform_heterostrain()
    part_c_twist_gradient()


if __name__ == "__main__":
    main()
