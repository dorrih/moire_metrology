# Contributing to moire-metrology

Thanks for your interest in contributing! This package is maintained by a
single researcher, so please be patient with review turnaround. Bug reports,
documentation fixes, new materials, and new examples are all welcome.

## Reporting issues

Before opening an issue, please search existing issues to avoid duplicates.

- **Bugs**: use the *Bug report* template and include a minimal reproducer,
  the package version (`python -c "import moire_metrology; print(moire_metrology.__version__)"`),
  Python version, and OS.
- **Feature requests**: use the *Feature request* template and explain the
  scientific use case — the package is small enough that scope decisions are
  driven by what real workflows need.
- **New materials**: use the *Add material* template (see the section below).

## Contributing code

This repo follows a standard fork-and-PR workflow:

1. **Fork** the repository on GitHub.
2. **Clone your fork** locally and add the upstream remote:
   ```bash
   git clone git@github.com:<your-username>/moire_metrology.git
   cd moire_metrology
   git remote add upstream git@github.com:dorrih/moire_metrology.git
   ```
3. **Create a feature branch** off `main`:
   ```bash
   git fetch upstream
   git checkout -b feature/my-change upstream/main
   ```
4. **Make your changes**, commit, and push to your fork.
5. **Open a pull request** against `dorrih/moire_metrology:main`. The PR
   template will walk you through the test plan checklist.

Direct pushes to `main` are blocked by branch protection — all changes go
through PRs, including the maintainer's own.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

The `[dev]` extra installs `pytest`, `pytest-cov`, `ruff`, and `build`.

### Running tests

```bash
pytest                # fast subset (default)
pytest -m slow        # slow integration tests (multi-layer relaxation)
pytest -m ""          # everything
```

The `slow` marker is excluded by default via `pyproject.toml`. CI runs both
the fast and slow subsets on every PR — please run at least the fast subset
locally before opening a PR.

### Linting

```bash
ruff check src/ tests/
```

CI will reject PRs that don't pass `ruff check`. Auto-fix where possible
with `ruff check --fix`.

## What goes where

- **`src/moire_metrology/`** — package source. Public API lives here. New
  features that change the public API should be discussed in an issue first.
- **`tests/`** — pytest tests. New code should come with tests; bug fixes
  should come with a regression test that fails before the fix.
- **`examples/`** — runnable scripts demonstrating end-to-end workflows.
  Examples must run end-to-end and should produce a plot or print a small
  summary; they are not part of the test suite but are spot-checked by the
  maintainer.
- **`docs_internal/`** — maintainer notes and design docs. Not part of the
  public package and not shipped in the sdist.

## Adding a new material or interface

`moire_metrology` separates two concerns:

- **`Material`** (`src/moire_metrology/materials.py`) — per-layer
  *intra-layer* properties: lattice constant and 2D elastic moduli.
- **`Interface`** (`src/moire_metrology/interfaces.py`) — the
  registry-dependent stacking energy *between two adjacent layers*.
  This is where GSFE lives.

GSFE is physically a property of the *pair* of materials in contact,
not of either material individually. Adding a new material (e.g. a
TMD that isn't bundled) and adding a new interface (e.g. a heterostack
of two existing materials) are separate contributions that can be
made together or independently.

### Adding a new `Material`

Each new entry needs:

1. **Name and chemical formula.**
2. **Lattice constant** (nm) with a citation.
3. **Bulk and shear moduli** (meV per unit cell) with a citation and
   an explicit note about the convention (we use the Carr/Zhou
   convention, meV/uc per layer).

That's it — `Material` does not carry GSFE.

### Adding a new `Interface`

Each new entry needs:

1. **Name** (e.g. `"MoSe2/WSe2 (H-stacked)"`).
2. **Bottom and top materials** — the two existing or newly-added
   `Material`s the interface couples.
3. **GSFE coefficients** `(c0, c1, c2, c3, c4, c5)` in meV/uc, in the
   Carr basis. For centrosymmetric homobilayers c4 = c5 = 0; for
   heterointerfaces with broken inversion symmetry they are non-zero.
4. **Stacking convention** — H-stacking, R-stacking, homobilayer, etc.,
   plus the `stacking_func` for homobilayer interfaces (or `None` for
   heterointerfaces, where no Bernal-like convention applies).
5. **Reference** — a citation string for the GSFE numbers (so the
   provenance travels with the data).
6. **A short test case** that exercises the new interface end-to-end
   (a small relaxation with an asserted energy or convergence
   behaviour).

Open an issue with the *Add material* or *Add interface* template
first so we can discuss the parameters and any convention questions
before you write the PR.

### Using a custom material/interface in your own scripts

If you just want to use a new material or interface in your own
scripts without upstreaming it, you can construct `Material` and
`Interface` instances directly — no code changes to the package
required. See the *Custom materials and interfaces* section of the
README for a complete example.

## Commit messages

- Use the imperative mood ("Add X", not "Added X" or "Adds X").
- First line ≤ 70 characters.
- Body explains *why*, not just *what*. The diff already shows the what.
- Reference issues with `#NN`.

## Reviewing and merging

The maintainer reviews and merges all PRs. CI must be green before merge:
ruff, fast tests on Python 3.10/3.11/3.12, slow tests, and the build smoke
test. PRs are merged with squash-and-merge by default to keep `main` linear.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By
participating you agree to abide by its terms.
