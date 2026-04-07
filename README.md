# moire-metrology

[![tests](https://img.shields.io/badge/tests-71%20passing-brightgreen)](#testing)
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

Alpha. The package is not yet on PyPI; install from source. The single-
interface relaxation, strain extraction, and pinning APIs are stable and
covered by 71 tests. Multi-layer relaxation is implemented at the energy /
gradient / Hessian level (and tested for correctness via finite differences),
but the solver path used in the original paper — implicit pseudo-time-stepping
— is not yet ported, so multi-layer convergence at very low twist angles is
known to be unreliable. See the issue tracker for the roadmap.

## Features

- **Single-interface relaxation** of twisted bilayer systems on a periodic
  triangular FEM mesh, with a damped Newton solver using the analytic
  elastic Hessian and a vertex-diagonal GSFE Hessian.
- **Materials database** with bundled GSFE parameterizations for graphene,
  hBN (AA, AA'), and graphene/hBN heterointerfaces, following the Carr et
  al. convention.
- **Strain extraction** from a measured moire pattern, implementing the
  closed-form `(λ₁, λ₂, φ₁, φ₂) → (θ, ε_c, ε_s)` inversion of
  Halbertal et al., ACS Nano (2022), including the spatially-varying
  registry-field extension.
- **Multi-layer stack API** (`LayerStack`) for heterostructures with
  multiple layers per flake. Energy / gradient / Hessian are correct and
  tested; see *Status* above for the solver caveat.
- **Constrained relaxation** with pinned stacking sites for working with
  partially-known experimental configurations.
- Plotting helpers for stacking-energy maps, elastic-energy maps, local
  twist angle, and displacement fields, with periodic tiling.

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

# Observed moire vectors (lengths in nm, angles in degrees)
result = get_strain_minimize_compression(
    alpha1=0.247, alpha2=0.247,
    lambda1=10.0, lambda2=10.0,
    phi1_deg=0.0, phi2_deg=60.0,
)
print(f"twist:    {result.theta_twist:.4f} deg")
print(f"eps_c:    {result.eps_c:.2e}")
print(f"eps_s:    {result.eps_s:.2e}")
```

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
.venv/bin/pytest                # fast subset (~10s, 64 tests)
.venv/bin/pytest -m slow        # integration tests (~8 min, 7 tests)
.venv/bin/pytest -m ""          # everything
```

The fast subset runs the algebraic correctness tests (gradient/Hessian
finite-difference checks, mesh, lattice, GSFE, strain extraction) on every
invocation. The slow tests do full relaxation solves and are gated behind
the `slow` marker so the dev loop stays tight.

## References

The methodology is described in the following papers:

- Halbertal et al., *Moiré metrology of energy landscapes in van der Waals
  heterostructures*, Nat. Commun. **12**, 242 (2021),
  [arXiv:2008.04835](https://arxiv.org/abs/2008.04835).
- Halbertal et al., *Multi-layered atomic relaxation in van der Waals
  heterostructures*,
  [arXiv:2206.06395](https://arxiv.org/abs/2206.06395).
- Halbertal, Shabani, Passupathy, Basov, *Extracting the Strain Matrix and
  Twist Angle from the Moiré Superlattice in van der Waals
  Heterostructures*, ACS Nano **16**, 1471 (2022),
  [doi:10.1021/acsnano.1c09789](https://pubs.acs.org/doi/10.1021/acsnano.1c09789).

If you use this package in published work, please cite the relevant
paper(s) above.

## License

MIT. See [LICENSE](LICENSE).
