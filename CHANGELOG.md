# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/dorrih/moire_metrology/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dorrih/moire_metrology/releases/tag/v0.1.0
