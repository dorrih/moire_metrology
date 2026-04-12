# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2026-04-12

### Added

- **`SolverConfig.rtol`** — relative gradient tolerance. Convergence is
  declared when `|grad| / |grad_initial| < rtol`. This criterion matters
  at low twist angles where the absolute gradient norm at the energy
  minimum can be O(1–10) due to large total energies, making any
  reasonable absolute `gtol` unreachable. Default: `1e-4`.
- **`SolverConfig.etol` and `SolverConfig.etol_window`** — energy
  stagnation tolerance and window size. If the fractional energy
  improvement over the last `etol_window` accepted steps falls below
  `etol`, the Newton solver declares convergence. Previously these were
  hard-coded internal constants; they are now user-configurable.
  Defaults: `etol=1e-6`, `etol_window=5`.
- **`RelaxationResult.converged`** property — boolean indicating whether
  the optimizer reported successful convergence.
- **`RelaxationResult.convergence_message`** property — human-readable
  description of the convergence outcome (e.g. which criterion was met,
  or "max iterations reached").
- **Post-hoc relative convergence check for L-BFGS-B.** SciPy's
  L-BFGS-B only supports an absolute `gtol`. The solver now captures the
  initial gradient norm and checks `rtol` after scipy returns, overriding
  the success flag when the relative criterion is met.
- Solver display output now includes the relative gradient ratio
  `(rel X.XXe-XX)` alongside the absolute gradient norm.
- Convergence exit messages now report which criterion triggered
  (absolute gtol, relative rtol, or energy stagnation) with numeric
  values.

### Fixed

- **`pseudo_dynamics` solver always reported failure even when
  converged.** The top-of-loop early exit (`if converged(gnorm): break`)
  fired after a single converged step, but the success flag required
  `count_conv >= 3` consecutive converged steps. The early break
  prevented `count_conv` from ever reaching 3, so the solver reported
  `success=False` on every run. Fixed by replacing the early break with
  `if count_conv >= count_conv_required` (#23).

## [0.5.0] - 2026-04-12

### Changed (breaking)

- **Three separate bilayer example scripts replaced by a single
  `bilayer_relaxation.py` with a `--preset` system.** The former
  `bilayer_graphene_relaxation.py`, `hbn_relaxation.py`, and
  `tmd_heterostructure.py` were identical except for default
  parameters.  The new script accepts `--preset graphene` (default),
  `--preset hbn`, or `--preset tmd`.  Any explicit CLI flag overrides
  the preset value.

### Added

- **CLI-configurable examples.** All bundled examples now accept
  `--interface` (bundled name or TOML file path), `--theta-twist`,
  `--pixel-size`, `--method`, `--max-iter`, `--gtol`, `--no-plots`,
  `--force`, and `--output-dir`.  The multilayer example adds
  `--n-top`, `--n-bottom`, and iterative-solver flags.  The strain
  extraction example adds `--heterostrain`, `--n-cells`, and
  `--skip-part-a` / `--skip-part-b`.
- **`--list-interfaces` flag** on every example.  Prints a detailed
  table of all bundled interfaces (materials, GSFE coefficients,
  literature references, CLI alias) and exits.
- **Shared `examples/_cli.py` helper** — interface name resolution
  with fuzzy matching (e.g. `graphene`, `mose2-wse2-h`, `hbn-aap`
  all work), TOML loading, graceful error messages for unsupported
  configurations (heterointerface on multilayer, missing TOML file,
  unknown interface name).
- **`docs/examples.md`** — guide to all bundled examples with preset
  tables, CLI argument reference, and usage recipes.
- **`docs/custom-materials.md`** — TOML schema reference for defining
  custom materials and interfaces, unit conventions, GSFE literature
  pointers, and limitation notes.
- **Example smoke tests in CI** — new `example smoke tests` job
  running `bilayer_relaxation.py --preset tmd`, Part A of the strain
  extraction example, `--list-interfaces`, and TOML interface loading.
  `ruff check` extended to cover `examples/`.

### Fixed

- **Duplicate `--list-interfaces` argparse registration** in
  `strain_extraction_and_pinning.py` and `spatial_strain_relaxation.py`
  caused an `ArgumentError` at parse time (PR #21).

## [0.4.1] - 2026-04-12

### Fixed

- **Spurious moire phase on homobilayer GSFE pairs.** When both layers
  of a homobilayer used the same Bernal stacking function, the solver
  injected an incorrect stacking offset that distorted the GSFE
  landscape. Fixed by detecting the homobilayer case and zeroing the
  intra-flake offset. Reproduces Fig 4 of Halbertal et al.
  arXiv:2206.06395 correctly (PR #18).

## [0.4.0] - 2026-04-10

### Changed (breaking)

- **`moire_metrology.strain.compute_strain_field` rewritten with the
  validated paper-Fig-1 formula.** The v0.3.0 scaffolding version took
  raw gradient arrays `(dIdx, dIdy, dJdx, dJdy)` and applied a closed-
  form `du/dx, du/dy` solve that was never validated against the paper.
  The new signature takes query coordinates and `RegistryField`
  instances directly, evaluates the analytic gradients internally, and
  recovers `(θ, ε_c, ε_s)` per point via the eq. 9 inversion of
  Halbertal et al. ACS Nano 16, 1471 (2022) followed by the per-point
  `get_strain` solver. Output keys are now `theta`, `eps_c`, `eps_s`
  (plus `lambda1`, `lambda2`, `phi1_deg`, `phi2_deg`) — matching what
  the paper figures plot. The old `eps_xx, eps_xy, eps_yy` keys are
  gone. Validated end-to-end against the maintainer's MATLAB
  spatial-extraction script and against paper Fig 1c-e on real
  H-MoSe2/WSe2 polyline data.
- **`compute_displacement_field` rewritten as a stacking-phase IC
  builder.** Same motivation as above — the v0.3.0 version was
  unvalidated scaffolding. The new signature takes a `MoireGeometry`
  and a `target_stacking` (one of `"AA"`, `"AB"`, `"BA"`) and solves
  `Mu @ u = (v0 - v_target)` per query point, where `v_target` comes
  from the polynomial registry fit plus the constant phase offset that
  puts integer registry sites at the requested stacking. Drops the
  unused `dr` parameter.
- **`FringeSet.fit_registry_fields` no longer resamples polylines.**
  The `resample_density` parameter is gone; the fit now uses raw
  polyline points only. Default `order` bumped from 8 to 11 to match
  the paper Methods section. The previous spline-resampling code path
  produced spurious 180° outliers at the data hull boundary on real
  data — it was never validated and is removed.
- **`theta0_deg` standardized to `phi0_deg`** across the spatial
  strain functions, matching the paper notation and the existing
  pointwise `get_strain(... phi0=...)` API.

### Added

- **`moire_metrology.strain.convex_hull_mask(data_x, data_y, qx, qy)`**
  helper for confining queries to the convex hull of a registry fit's
  training data. A high-degree polynomial extrapolates with rapid
  growth outside its data support; this mask catches mesh vertices and
  grid points that lie past the data extent before they produce
  nonsense displacements or strain values.
- **`tests/test_strain_spatial.py`** — 9 unit tests covering the
  rewritten spatial strain API: rigid-twist registry → recovered
  `|θ|` exact and zero strain; output shape preservation; the IC
  realizes the `v_target` phase contract via `geom.stacking_phases`;
  switching `target_stacking` adds a constant displacement offset;
  unknown stacking → `ValueError`; convex hull classification.
- **`compute_strain_field` now also returns the full strain tensor.**
  Output dict gains `S11`, `S12`, `S22` (symmetric strain tensor in
  the global `(x, y)` frame) plus `eps1`, `eps2`, `strain_angle`
  (principal strains and axis). The previously returned `theta`,
  `eps_c`, `eps_s`, `lambda*`, `phi*_deg` keys are unchanged. The
  new components are the natural inputs to a gradient-integration
  IC builder for the relaxation framework.
- **`displacement_from_strain_field(disc, theta_deg, theta_avg_deg,
  S11, S12, S22, pin_vertex)`** — new helper that integrates a target
  local twist + strain field on the FEM mesh into a displacement field
  `u(r)` in the relaxation framework's native language. Implementation
  is a sparse least-squares Poisson reconstruction:
  `[Dx; Dy] @ u_x = (S11; S12 - δθ)` and similarly for `u_y`, with
  one vertex pinned to fix the global translation gauge. The eps=0.5
  layer-partition factor was confirmed against `energy.py` /
  `discretization.py` (factor of 1 on both rotation and strain).
  Where the previous pointwise phase-matching IC built a step
  discontinuity in `u` at the data hull boundary (cliff that the
  elastic energy hated), the gradient-integration IC has continuous
  `u(r)` everywhere.
- **`examples/spatial_strain_extraction.py` → `examples/spatial_strain_relaxation.py`,
  re-extended with the relaxation step.** The example now uses
  `displacement_from_strain_field` to build the IC from the strain
  extraction output (with the strain field zeroed outside the data
  convex hull, so the integrated `u` smoothly relaxes to the average
  configuration there), then runs Newton relaxation against
  `MOSE2_WSE2_H_INTERFACE` with three-sublattice pinning (AA + BA +
  AB sites from the polynomial registry fit). Headline figure has 4
  panels: paper Fig 1c-e plus the relaxed stacking-energy density
  showing the equilibrium domain pattern aligned with the traced
  polylines.
- **`examples/hbn_relaxation.py`** — bundled example demonstrating
  the graphene/hBN heterointerface relaxation. Headline case is
  θ = 0° (pure lattice-mismatch moiré, λ ≈ 15.75 nm), where the
  1.6% intrinsic mismatch between graphene and hBN drives a moiré
  pattern even without twist. The relaxed stacking-energy map shows
  the textbook single-minimum hexagonal domain pattern (vs the AB/BA
  triangular pattern of TBG). Runs end-to-end in ~3 seconds with
  L-BFGS-B. Uses `GRAPHENE_HBN_INTERFACE` and the new v0.3.0
  `Material.moduli_n_per_m` property to print literature-traceable
  N/m values for the materials at run start.
- **`examples/tmd_heterostructure.py`** — bundled example
  demonstrating the H-stacked MoSe2/WSe2 heterointerface at θ = 1.5°
  (λ ≈ 12.5 nm). The relaxed stacking-energy map shows the three
  distinct stacking minima (XX', MX', MM') from the broken
  inversion symmetry, and the energy reduction is ~34% — much
  larger than TBG's ~30%, the textbook signature of a "deep moiré
  potential". Runs end-to-end in ~30 seconds with L-BFGS-B. Cites
  Shabani / Halbertal Nat. Phys. 17, 720 (2021).
- README *Quick start* section now points at all three bundled
  example scripts (graphene + hBN + TMD).
- **Modified-Hessian Newton solver** — flips negative eigenvalues of
  the per-vertex 2×2 GSFE Hessian blocks before assembly. The GSFE
  curvature at AA sites has eigenvalues down to -3.9×10⁶; the old
  scalar Levenberg-Marquardt damping couldn't compensate efficiently.
  The per-vertex flip produces a globally PD Hessian, enabling
  meaningful Newton steps from iteration 1. Converges 3× faster than
  `pseudo_dynamics` in the early phase.
- **3 new round-trip tests** in `tests/test_strain_spatial.py` for
  `displacement_from_strain_field`: pure strain, pure rotation, and
  general affine `u`. The integrator reconstructs the target field
  exactly (up to the global translation gauge), pinning down the
  partition-factor convention concretely.

## [0.3.0] - 2026-04-08

### Added

- **TOML loaders for `Material` and `Interface`.** Both dataclasses
  now have `from_dict()` and `from_toml()` classmethods so users can
  define a custom material or heterointerface in a standalone TOML
  file and load it without forking the package. The schema mirrors
  the dataclass field names; an `[interface]` table inlines its two
  materials under `[interface.bottom]` and `[interface.top]`. Worked
  example at `examples/data/mose2_wse2_h.toml` reproduces the bundled
  `MOSE2_WSE2_H_INTERFACE` and is asserted bit-identical in the test
  suite. `tomli>=2.0` added as a conditional dependency for Python
  3.10; `tomllib` (stdlib) is used on 3.11+.
- README *Custom materials and interfaces* section now has a TOML
  loader subsection alongside the existing direct-construction one.
- 16 new tests in `tests/test_toml_loader.py`: round-trips, missing
  fields, typo'd extra fields (with a helpful "did you mean
  Interface?" pointer for GSFE on `Material`), wrong-length GSFE
  coefficients, non-numeric GSFE coefficients, and end-to-end
  numerical equivalence to the bundled equivalent.
- **`Material.from_2d_moduli_n_per_m()` constructor** — accepts K, G
  in literature-standard 2D N/m units (the convention used in
  Lee et al. Science 2008 and most experimental papers) and converts
  to the internal meV/uc convention. Lets future custom materials be
  specified in their natural units without the user having to redo
  the conversion arithmetic by hand.
- **`Material.moduli_n_per_m` property** — round-trip helper that
  returns `(K, G)` in N/m for sanity-checking existing materials
  against published indentation values.
- 9 new tests in `tests/test_elastic_units.py` covering the converter
  round-trip, the lattice-constant scaling, and regression-locking
  the corrected `GRAPHENE` parameters against the paper values.

### Fixed

- **`pyproject.toml` version bumped from `"0.1.0"` to `"0.2.0"`.**
  The v0.2.0 refactor PR bumped `__version__` in
  `src/moire_metrology/__init__.py` but missed the wheel metadata in
  `pyproject.toml`, so the wheel and sdist built from the v0.2.0 tag
  still self-identify as `0.1.0`. The Zenodo archive of the v0.2.0
  tag is frozen with the wrong wheel metadata; this fix takes effect
  for the next release tag (v0.2.1 / v0.3.0).
- **Graphene elastic moduli were wrong by a factor of ~8.** v0.1.0
  and v0.2.0 shipped `GRAPHENE.bulk_modulus = 8595 meV/uc` and
  `shear_modulus = 5765 meV/uc`, which round-trip to ~26 N/m and
  ~17 N/m — far below the experimental literature value of ~211 N/m
  (Lee et al. Science 321, 385 (2008)). The values were not from any
  cited source and did not match the published Halbertal et al.
  Nat. Commun. 12, 242 (2021) SI Table 1, which gives K=69518,
  G=47352 meV/uc citing Carr et al. PRB 98, 224102 (2018). This
  commit restores the paper values verbatim. **Any quantitative
  graphene relaxation result from a previous version was using ~8×
  too soft elastic constants and is biased toward over-relaxation
  (domain walls too narrow, AA areas too small).**
- **Graphene GSFE coefficients were ~3% drifted from the paper.**
  The package derived them from a Zhou et al. starting point via
  `_zhou_to_carr` and produced `(7.036, 4.041, -0.372, -0.094, 0, 0)`,
  whereas the paper Table 1 gives `(6.832, 4.064, -0.374, -0.095, 0, 0)`
  taken directly from Carr et al. with no transformation. This commit
  hard-codes the literal Carr/paper values for graphene, eliminating
  the drift.

- **hBN elastic moduli now from Falin et al. Nat. Commun. 8, 15815
  (2017).** v0.1.0 and v0.2.0 shipped `HBN_AA.bulk_modulus = 8595` and
  `shear_modulus = 5765 meV/uc` — the same wrong-by-~8x values graphene
  had, copy-pasted from the pre-fix `GRAPHENE` entry, with no hBN
  source. They round-tripped to ~26 / ~17 N/m, which is physically
  implausible for hBN. This commit replaces them with the literature-
  derived values: Falin et al. report `E_3D = 0.865 ± 0.073 TPa` for
  monolayer hBN by indentation, which gives `E_2D ≈ 286 N/m` using a
  0.334 nm thickness. With `ν ≈ 0.21` (also Falin), the standard
  isotropic 2D Lamé relations give `K_2D ≈ 181 N/m, G_2D ≈ 118 N/m`,
  which the new `from_2d_moduli_n_per_m` constructor converts to
  `K ≈ 61638 meV/uc, G ≈ 40252 meV/uc`. Both `HBN_AA` and `HBN_AAP`
  now use these values; the AA / AA' designation is a stacking
  convention for the *interface*, not a per-layer material property.
- **All bundled materials and interfaces now have explicit literature
  citations** as inline source comments AND as `reference=` strings on
  the `Interface` entries. The hBN GSFE coefficients on
  `HBN_AA_HOMOBILAYER`, `HBN_AAP_HOMOBILAYER`, and
  `GRAPHENE_HBN_INTERFACE` were verified to be exact verbatim copies
  of Zhou et al. PRB 92, 155438 (2015), Table III, by reading
  `docs_internal/zhou2015.pdf` directly. The graphene K, G and GSFE
  on `GRAPHENE` and `GRAPHENE_GRAPHENE` were verified against Carr et
  al. PRB 98, 224102 (2018), Table I, by reading
  `docs_internal/carr2018.pdf` directly. The MoSe2/WSe2 entries were
  already verified against the Halbertal Nat. Commun. 2021 SI Table 1
  and Shabani Nat. Phys. 2021 Methods section in the v0.2.0 work.
  **The repo no longer contains any uncited material/interface
  parameters.**

## [0.2.0] - 2026-04-08

### Changed (breaking)

- **GSFE moved off `Material` and onto a new `Interface` dataclass.**
  GSFE describes the registry-dependent stacking energy *between two
  adjacent layers*, not a property of either material individually.
  The v0.1.0 model conflated the two and worked around it for the
  bundled `MOSE2`/`WSE2` heterostructure by storing duplicate
  coefficients on both materials. The new model has a clean
  separation: `Material` carries `name`, `lattice_constant`,
  `bulk_modulus`, `shear_modulus`; `Interface` carries `name`, `bottom`,
  `top`, `gsfe_coeffs`, `stacking_func`, `reference`.

- **`RelaxationSolver.solve()` API**: replace
  ```python
  solver.solve(material1=GRAPHENE, material2=GRAPHENE, theta_twist=1.05)
  ```
  with
  ```python
  from moire_metrology.interfaces import GRAPHENE_GRAPHENE
  solver.solve(moire_interface=GRAPHENE_GRAPHENE, theta_twist=1.05)
  ```
  For multi-layer flakes also pass `top_interface=` and/or
  `bottom_interface=`, and use `n_top` / `n_bottom` instead of
  `nlayer1` / `nlayer2`. The legacy kwargs raise a `TypeError` with a
  redirect to the new signature.

- **`LayerStack`**: same migration. `LayerStack(top=GRAPHENE, n_top=2,
  bottom=GRAPHENE, n_bottom=3, theta_twist=...)` becomes
  `LayerStack(moire_interface=GRAPHENE_GRAPHENE,
  top_interface=GRAPHENE_GRAPHENE, bottom_interface=GRAPHENE_GRAPHENE,
  n_top=2, n_bottom=3, theta_twist=...)`. Validation is done at
  construction time: missing or mismatched homobilayer interfaces raise
  `ValueError` with a clear pointer to the offending field.

- **`RelaxationResult`**: gains `moire_interface`, `top_interface`,
  `bottom_interface` fields and drops the standalone `material1`,
  `material2` fields. `result.material1` and `result.material2` remain
  available as convenience properties pointing at
  `moire_interface.top` and `moire_interface.bottom`. The cached
  `.npz` schema bumps to include `moire_interface_name`; old caches
  from v0.1.0 need to be regenerated.

- **`GRAPHENE_ON_HBN` material removed.** The graphene-on-hBN
  heterostructure is now expressed as the `GRAPHENE_HBN_INTERFACE`
  bundled interface that pairs the standalone `GRAPHENE` and `HBN_AA`
  materials. Note: the hBN polytype the underlying Zhou et al. GSFE
  was fitted against is still pending verification — see the inline
  TODO in `interfaces.py`.

### Added

- New `moire_metrology.interfaces` submodule with the `Interface`
  dataclass and bundled entries: `GRAPHENE_GRAPHENE`,
  `HBN_AA_HOMOBILAYER`, `HBN_AAP_HOMOBILAYER`,
  `GRAPHENE_HBN_INTERFACE`, `MOSE2_WSE2_H_INTERFACE`. Each entry
  carries a `reference` string with the literature citation for its
  GSFE numbers.
- `BUNDLED_INTERFACES` tuple containing every bundled interface.
- `Interface.is_homobilayer` property.
- README "Custom materials and interfaces" section showing how to
  construct user-defined `Material` and `Interface` objects without
  forking the package.
- New tests: legacy-kwargs error path, custom-Interface end-to-end
  agreement with bundled equivalent.

### Fixed

- v0.1.0 always sourced the moiré interface GSFE from
  `material1.gsfe_coeffs`, ignoring `material2.gsfe_coeffs`. For
  user-defined heterointerfaces this could silently produce wrong
  physics. The new model makes the moiré GSFE come from the explicit
  `moire_interface=` argument, eliminating the bug at the API
  boundary.

## [0.1.0] - 2026-04-08

Initial public release.

### Added

- Core moire relaxation solver (Phase 1).
- Strain extraction from moire patterns (Phase 2).
- Multi-layer relaxation (Phase 3), including Hessian finite-difference
  tests for multi-layer configurations.
- Constrained relaxation with pinned stacking sites.
- `pseudo_dynamics` solver: implicit theta-method on the gradient flow,
  with a matrix-free MINRES path for large systems (#1, #4).
- `fix_top` / `fix_bottom` layer-clamping options on `LayerStack` and the
  solver (#1).
- `examples/multilayer_penetration.py`: bulk-uniform substrate relaxation
  example (#3).
- Finite-mesh / point-pinning relaxation support for non-periodic geometries
  (Plan B MVP) (#5).
- Public-release scaffolding: `LICENSE` (MIT), `README.md`, full
  `pyproject.toml` metadata, GitHub Actions CI (ruff + tests on Python
  3.10/3.11/3.12 + sdist/wheel build smoke test), `CONTRIBUTING.md`,
  issue and PR templates, citation file.

### Fixed

- Strain extraction: alpha double-counting in the deformation matrix (#6).
- Various bug fixes and example/README polish from the hardening pass.

[Unreleased]: https://github.com/dorrih/moire_metrology/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/dorrih/moire_metrology/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/dorrih/moire_metrology/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/dorrih/moire_metrology/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/dorrih/moire_metrology/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/dorrih/moire_metrology/releases/tag/v0.3.0
[0.2.0]: https://github.com/dorrih/moire_metrology/releases/tag/v0.2.0
[0.1.0]: https://github.com/dorrih/moire_metrology/releases/tag/v0.1.0
