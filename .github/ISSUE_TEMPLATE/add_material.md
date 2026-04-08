---
name: Add material or interface
about: Propose a new built-in Material and/or Interface for moire_metrology
title: "[MATERIAL] Add <material or interface name>"
labels: enhancement, material
---

`moire_metrology` separates `Material` (per-layer elastic properties)
from `Interface` (the GSFE coupling between two adjacent layers).
Built-in materials live in `src/moire_metrology/materials.py`; built-in
interfaces live in `src/moire_metrology/interfaces.py`. Please open
this issue *before* writing the PR so we can discuss conventions and
parameters first.

If you just want to use a new material or interface in your own scripts
without upstreaming it, the README's *Custom materials and interfaces*
section shows how to construct `Material` and `Interface` instances
directly — no code changes required.

Fill in the section(s) relevant to your contribution. A new material
without a new interface is fine (just leave the *Interface* section
blank), and a new interface between two existing materials is fine
too (just leave the *Material* section blank).

## Material

Name, chemical formula, common abbreviation:

## Lattice constant

- Value (nm):
- Citation:

## Elastic moduli

The package uses the Carr/Zhou convention (meV per unit cell).

- Bulk modulus (meV/uc):
- Shear modulus (meV/uc):
- Citation:
- Notes on convention (especially if the source uses different units):

## Interface

(Skip this section if you're only adding a new `Material` and
not a new `Interface`.)

- Interface name (e.g. `MoSe2/WSe2 (H)`, `Graphene/hBN`):
- Bottom material:
- Top material:
- Stacking convention — homobilayer, H-stacking, R-stacking, etc., and
  any sign or origin choices that matter for reproducing the GSFE
  landscape:

### GSFE coefficients

Generalized stacking fault energy expansion `(c0, c1, c2, c3, c4, c5)`
in meV/uc, in the Carr basis. For centrosymmetric homobilayers
c4 = c5 = 0; for heterointerfaces with broken inversion symmetry they
are non-zero.

- `c0`:
- `c1`:
- `c2`:
- `c3`:
- `c4`:
- `c5`:
- Citation:
- Basis convention (especially if the source uses a different Fourier
  basis and a transformation is needed):

## Test case

A short end-to-end test (small relaxation, asserted energy or
convergence behaviour) that exercises the new material/interface.
Sketch what you'd assert.

## Additional notes

Anything else relevant — alternative parameter sets in the literature,
known caveats, temperature dependence, etc.
