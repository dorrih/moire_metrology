# Installation

## From PyPI

```bash
pip install moire-metrology
```

Requires Python 3.10+. The package depends on `numpy`, `scipy`, and
`matplotlib`.

## From source (development)

```bash
git clone https://github.com/dorrih/moire_metrology.git
cd moire_metrology
python3.11 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

The `[dev]` extra adds `pytest`, `pytest-cov`, `ruff`, and `build`.

## Optional dependencies

- **meshpy** -- required only for `generate_finite_mesh()` (non-periodic
  triangulated meshes). Install with `pip install moire-metrology[meshpy]`.

## Verifying the installation

```python
import moire_metrology
print(moire_metrology.__version__)
```
