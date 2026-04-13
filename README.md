# moire-metrology

[![ci](https://github.com/dorrih/moire_metrology/actions/workflows/ci.yml/badge.svg)](https://github.com/dorrih/moire_metrology/actions/workflows/ci.yml)
[![docs](https://github.com/dorrih/moire_metrology/actions/workflows/docs.yml/badge.svg)](https://dorrih.github.io/moire_metrology/)
[![status](https://img.shields.io/badge/status-beta-blue)](#status)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](#)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19468557.svg)](https://doi.org/10.5281/zenodo.19468557)

Atomic relaxation, strain extraction, and multi-layer analysis for twisted
2D van der Waals heterostructures.

This is a Python re-implementation of the modeling tools developed for the
papers listed under [References](#references). The goal is to make the
methodology accessible as a pip-installable package, instead of a private
collection of MATLAB scripts.

**[Documentation](https://dorrih.github.io/moire_metrology/)** --
full user guide, theory background, API reference, and examples.

## Status

Beta. The single-interface relaxation, strain extraction, multi-layer
stack, and finite-mesh point-pinning APIs are all in place and covered
by 70+ fast tests plus integration tests gated behind a `slow` marker.

## Features

- **Single-interface relaxation** of twisted bilayer systems on a
  periodic triangular FEM mesh, with three solver options: damped Newton,
  implicit pseudo-time-stepping, and L-BFGS-B.
- **Multi-layer stack API** (`LayerStack`) for heterostructures with
  any number of layers per flake, including `fix_top` / `fix_bottom`
  clamps to approximate semi-infinite substrates.
- **Materials database** with bundled GSFE parameterizations for
  graphene, hBN (AA, AA'), MoSe2/WSe2, and graphene/hBN heterointerfaces.
- **Strain extraction** from measured moire patterns, implementing the
  closed-form inversion of Halbertal et al., ACS Nano (2022).
- **Constrained relaxation on a finite domain** via `PinningMap`.
- Plotting helpers for stacking-energy maps, elastic-energy maps,
  local twist angle, and displacement fields.

## Installation

```bash
pip install moire-metrology
```

Or install from source for development:

```bash
git clone https://github.com/dorrih/moire_metrology.git
cd moire_metrology
python3.11 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

Requires Python 3.10+. Dependencies: `numpy`, `scipy`, `matplotlib`.

## Quick start

```python
from moire_metrology import GRAPHENE_GRAPHENE, RelaxationSolver, SolverConfig

solver = RelaxationSolver(SolverConfig(pixel_size=1.0, max_iter=200))
result = solver.solve(
    moire_interface=GRAPHENE_GRAPHENE,
    theta_twist=1.05,  # degrees
)

print(f"Moire wavelength: {result.geometry.wavelength:.1f} nm")
print(f"Energy reduction: {100 * result.energy_reduction:.1f}%")

result.plot_stacking(n_tile=2)        # AB/BA triangular domains
result.plot_elastic_energy(n_tile=2)  # SDW network
result.plot_local_twist(n_tile=2)     # AA vortex map
```

All bundled examples are CLI-configurable:

```bash
python examples/bilayer_relaxation.py                    # TBG default
python examples/bilayer_relaxation.py --preset hbn       # graphene/hBN
python examples/bilayer_relaxation.py --preset tmd       # MoSe2/WSe2
python examples/bilayer_relaxation.py --list-interfaces  # show all bundled interfaces
```

See the [documentation](https://dorrih.github.io/moire_metrology/) for
the full quick-start guide, multi-layer stacks, strain extraction,
constrained relaxation, custom materials via TOML, and the API reference.

## References

| Capability | Reference |
|---|---|
| Single-interface relaxation, GSFE-based continuum model | [1] |
| Multi-layer (LayerStack) relaxation | [2] |
| Strain matrix and twist-angle extraction from moire vectors | [3] |
| Spatially-varying / point-pinned constrained relaxation | [4] |

1. Halbertal, D. *et al.* **Moire metrology of energy landscapes in van der
   Waals heterostructures.** *Nat. Commun.* **12**, 242 (2021).
   [doi:10.1038/s41467-020-20428-1](https://www.nature.com/articles/s41467-020-20428-1)
   -- [arXiv:2008.04835](https://arxiv.org/abs/2008.04835)

2. Halbertal, D. *et al.* **Multilayered Atomic Relaxation in van der Waals
   Heterostructures.** *Phys. Rev. X* **13**, 011026 (2023).
   [doi:10.1103/PhysRevX.13.011026](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.13.011026)
   -- [arXiv:2206.06395](https://arxiv.org/abs/2206.06395)

3. Halbertal, D., Shabani, S., Pasupathy, A. N. & Basov, D. N.
   **Extracting the Strain Matrix and Twist Angle from the Moire
   Superlattice in van der Waals Heterostructures.**
   *ACS Nano* **16**, 1471--1476 (2022).
   [doi:10.1021/acsnano.1c09789](https://pubs.acs.org/doi/10.1021/acsnano.1c09789)

4. Shabani, S., Halbertal, D., Wu, W., Chen, M., Liu, S., Hone, J.,
   Yao, W., Basov, D. N., Zhu, X. & Pasupathy, A. N. **Deep moire
   potentials in twisted transition metal dichalcogenide bilayers.**
   *Nature Physics* **17**, 720--725 (2021).
   [doi:10.1038/s41567-021-01174-7](https://www.nature.com/articles/s41567-021-01174-7)
   -- [arXiv:2008.07696](https://arxiv.org/abs/2008.07696)

### Citing the software itself

The package is archived on Zenodo with a concept DOI that always resolves
to the latest version:

- **DOI (always-latest):** [10.5281/zenodo.19468557](https://doi.org/10.5281/zenodo.19468557)

A machine-readable citation file is provided as [`CITATION.cff`](CITATION.cff);
GitHub surfaces it via the "Cite this repository" button.

## License

MIT. See [LICENSE](LICENSE).
