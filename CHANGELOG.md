# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
