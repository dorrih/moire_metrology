# moire-metrology

Atomic relaxation, strain extraction, and multi-layer analysis for twisted
2D van der Waals heterostructures.

This is a Python re-implementation of the modeling tools developed for the
papers listed under {doc}`references`. The goal is to make the methodology
accessible as a pip-installable package, instead of a private collection
of MATLAB scripts.

## Features

- **Single-interface relaxation** of twisted bilayer systems on a periodic
  triangular FEM mesh, with three solver options: damped Newton (analytic
  Hessian), implicit pseudo-time-stepping, and L-BFGS-B.
- **Multi-layer stack API** (`LayerStack`) for heterostructures with any
  number of layers per flake, including `fix_top` / `fix_bottom` clamps.
- **Materials database** with bundled GSFE parameterizations for graphene,
  hBN (AA, AA'), MoSe2/WSe2, and graphene/hBN heterointerfaces.
- **Strain extraction** from measured moire patterns, implementing the
  closed-form inversion of Halbertal et al., ACS Nano (2022).
- **Constrained relaxation on a finite domain** via `PinningMap` -- pin
  known stacking configurations and relax the rest.
- Plotting helpers for stacking-energy maps, elastic-energy maps, local
  twist angle, and displacement fields.

```{toctree}
:maxdepth: 2
:caption: User Guide

installation
quickstart
theory
examples
custom-materials
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api
```

```{toctree}
:maxdepth: 1
:caption: Project

references
changelog
```
