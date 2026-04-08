# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
