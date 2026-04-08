# moire-metrology

[![ci](https://github.com/dorrih/moire_metrology/actions/workflows/ci.yml/badge.svg)](https://github.com/dorrih/moire_metrology/actions/workflows/ci.yml)
[![status](https://img.shields.io/badge/status-alpha-orange)](#status)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](#)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Atomic relaxation, strain extraction, and multi-layer analysis for twisted
2D van der Waals heterostructures.

This is a Python re-implementation of the modeling tools developed for the
papers listed under [References](#references). The goal is to make the
methodology accessible as a pip-installable package, instead of a private
collection of MATLAB scripts.

## Status

Alpha. The package is not yet on PyPI; install from source. The
single-interface relaxation, strain extraction, multi-layer stack, and
finite-mesh point-pinning APIs are all in place and covered by 70+ fast
tests plus integration tests gated behind a `slow` marker. The
implicit pseudo-time-stepping solver from the paper MATLAB code has
been ported and is the recommended choice for stiff multi-layer / low
twist angle problems; for routine bilayer work the damped Newton path
is usually faster.

## Features

- **Single-interface relaxation** of twisted bilayer systems on a
  periodic triangular FEM mesh, with three solver options: a damped
  Newton method using the analytic elastic + GSFE Hessians, an
  implicit pseudo-time-stepping (theta-method) solver ported from the
  paper code (recommended for stiff multi-layer cases), and an
  L-BFGS-B fallback. The pseudo_dynamics solver also has an opt-in
  matrix-free MINRES linear-solve path for large meshes.
- **Multi-layer stack API** (`LayerStack`) for heterostructures with
  any number of layers per flake, including optional `fix_top` /
  `fix_bottom` clamps to approximate semi-infinite substrates.
- **Materials database** with bundled GSFE parameterizations for
  graphene, hBN (AA, AA'), and graphene/hBN heterointerfaces, following
  the Carr et al. convention.
- **Strain extraction** from a measured moire pattern, implementing the
  closed-form `(λ₁, λ₂, φ₁, φ₂) → (θ, ε_c, ε_s)` inversion of
  Halbertal et al., ACS Nano (2022), with both pointwise and
  spatially-varying registry-field extensions.
- **Constrained relaxation on a finite domain** via `PinningMap` —
  user pins selected stacking configurations at chosen positions in
  an experimental image, the relaxation fills in the rest. Works on
  both periodic moire-cell meshes and finite (non-wrapping) meshes
  produced by `generate_finite_mesh`. This is the workflow analogous
  to the spatially-varying-strain reconstruction in Shabani, Halbertal
  et al., *Nat. Phys.* (2021).
- Plotting helpers for stacking-energy maps, elastic-energy maps,
  local twist angle, and displacement fields, with periodic tiling
  and automatic filtering of wrap-around triangles.

## Installation

```bash
git clone <this-repo>
cd moire_metrology
python3.11 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

Requires Python 3.10+. The package depends on `numpy`, `scipy`, and
`matplotlib`. The `[dev]` extra adds `pytest`, `pytest-cov`, and `ruff`.

## Quick start

```python
from moire_metrology import GRAPHENE, RelaxationSolver, SolverConfig

solver = RelaxationSolver(SolverConfig(pixel_size=1.0, max_iter=200))
result = solver.solve(
    material1=GRAPHENE,
    material2=GRAPHENE,
    theta_twist=1.05,  # degrees
)

print(f"Moire wavelength: {result.geometry.wavelength:.1f} nm")
print(f"Energy reduction: {100 * result.energy_reduction:.1f}%")

result.plot_stacking(n_tile=2)        # AB/BA triangular domains
result.plot_elastic_energy(n_tile=2)  # SDW network
result.plot_local_twist(n_tile=2)     # AA vortex map
```

For a complete worked example reproducing the hallmark TBG relaxation
pattern (low twist, AA vortices over AB/BA domains), see
[`examples/bilayer_graphene_relaxation.py`](examples/bilayer_graphene_relaxation.py).
The example caches the relaxed state to a `.npz` so you can iterate on plots
without re-solving.

### Strain extraction from a measured moire

```python
from moire_metrology.strain import get_strain_minimize_compression
from moire_metrology.strain.extraction import shear_strain_invariant

# Observed moire vectors (lengths in nm, angles in degrees)
result = get_strain_minimize_compression(
    alpha1=0.247, alpha2=0.247,
    lambda1=10.0, lambda2=10.0,
    phi1_deg=0.0, phi2_deg=60.0,
)
print(f"twist:           {result.theta_twist:.4f} deg")
print(f"compression ε_c: {result.eps_c:.2e}")

# For the shear strain ε_s, prefer the closed-form invariant helper
# (the result.eps_s field has a known φ₀-dependence bug; see issues).
eps_s = shear_strain_invariant(0.247, 0.247, 10.0, 10.0, 0.0, 60.0)
print(f"shear ε_s:       {eps_s:.2e}")
```

For a complete worked example sweeping `Δφ` and reproducing
Fig. 3 of the ACS Nano paper, see
[`examples/strain_extraction_and_pinning.py`](examples/strain_extraction_and_pinning.py).

### Constrained relaxation on a finite domain (pinned stacking sites)

```python
from moire_metrology import GRAPHENE, RelaxationSolver, SolverConfig
from moire_metrology.lattice import HexagonalLattice, MoireGeometry
from moire_metrology.mesh import generate_finite_mesh
from moire_metrology.discretization import Discretization
from moire_metrology.pinning import PinningMap

# Build a finite (non-periodic) mesh covering several moire cells
lat = HexagonalLattice(alpha=GRAPHENE.lattice_constant)
geom = MoireGeometry(lat, theta_twist=2.0)
mesh = generate_finite_mesh(geom, n_cells=4, pixel_size=0.7)

# Pin a few interior points to known stacking configurations
pins = PinningMap(mesh, geom)
pins.pin_stacking(x=10.0, y=10.0, stacking="AA", radius=0.8)
pins.pin_stacking(x=20.0, y=15.0, stacking="AB", radius=0.8)

disc = Discretization(mesh, geom)
conv = disc.build_conversion_matrices(nlayer1=1, nlayer2=1)
constraints = pins.build_constraints(conv)

# Run the relaxation on the finite mesh with the pin constraints
result = RelaxationSolver(SolverConfig(method="L-BFGS-B")).solve(
    material1=GRAPHENE, material2=GRAPHENE, theta_twist=2.0,
    mesh=mesh, constraints=constraints,
)
```

This is the spatially-varying-strain workflow used in
Shabani / Halbertal et al. (Fig. 1h of the *Nat. Phys.* paper)
to reconstruct the relaxed stacking-energy density of an experimental
STM topograph from a small set of identified stacking sites. See the
[example script](examples/strain_extraction_and_pinning.py) for the
end-to-end version.

### Multi-layer stack

```python
from moire_metrology import GRAPHENE, SolverConfig
from moire_metrology.multilayer import LayerStack

stack = LayerStack(
    top=GRAPHENE, n_top=1,
    bottom=GRAPHENE, n_bottom=2,
    theta_twist=1.5,
)
result = stack.solve(SolverConfig(method="L-BFGS-B", pixel_size=1.0))
```

## Testing

```bash
.venv/bin/pytest                # fast subset (~30 s, 70+ tests)
.venv/bin/pytest -m slow        # integration tests (~7 min, 7 tests)
.venv/bin/pytest -m ""          # everything
```

The fast subset runs the algebraic correctness tests (gradient/Hessian
finite-difference checks, mesh, lattice, GSFE, strain extraction, finite-
mesh discretization, pinned-constraint round-trips) on every invocation.
The slow tests do full relaxation solves and are gated behind the `slow`
marker so the dev loop stays tight.

## References

The methodology implemented here is described in the following peer-reviewed
papers. Each capability of the package is grounded in one (or more) of them;
if you use this package in published work, please cite the relevant paper(s).

| Capability | Reference |
|---|---|
| Single-interface relaxation, GSFE-based continuum model | [1] |
| Multi-layer (LayerStack) relaxation | [2] |
| Strain matrix and twist-angle extraction from moire vectors | [3] |
| Spatially-varying / point-pinned constrained relaxation | [4] |

1. Halbertal, D. *et al.* **Moiré metrology of energy landscapes in van der
   Waals heterostructures.** *Nat. Commun.* **12**, 242 (2021).
   [doi:10.1038/s41467-020-20428-1](https://www.nature.com/articles/s41467-020-20428-1)
   · [arXiv:2008.04835](https://arxiv.org/abs/2008.04835)

2. Halbertal, D. *et al.* **Multilayered Atomic Relaxation in van der Waals
   Heterostructures.** *Phys. Rev. X* **13**, 011026 (2023).
   [doi:10.1103/PhysRevX.13.011026](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.13.011026)
   · [arXiv:2206.06395](https://arxiv.org/abs/2206.06395)

3. Halbertal, D., Shabani, S., Pasupathy, A. N. & Basov, D. N.
   **Extracting the Strain Matrix and Twist Angle from the Moiré
   Superlattice in van der Waals Heterostructures.**
   *ACS Nano* **16**, 1471–1476 (2022).
   [doi:10.1021/acsnano.1c09789](https://pubs.acs.org/doi/10.1021/acsnano.1c09789)

4. Shabani, S., Halbertal, D., Wu, W., Chen, M., Liu, S., Hone, J.,
   Yao, W., Basov, D. N., Zhu, X. & Pasupathy, A. N. **Deep moiré
   potentials in twisted transition metal dichalcogenide bilayers.**
   *Nature Physics* **17**, 720–725 (2021).
   [doi:10.1038/s41567-021-01174-7](https://www.nature.com/articles/s41567-021-01174-7)
   · [arXiv:2008.07696](https://arxiv.org/abs/2008.07696)

A machine-readable citation file is also provided as
[`CITATION.cff`](CITATION.cff).

## License

MIT. See [LICENSE](LICENSE).
