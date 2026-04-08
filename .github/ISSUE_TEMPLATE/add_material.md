---
name: Add material
about: Propose a new built-in material for moire_metrology.materials
title: "[MATERIAL] Add <material name>"
labels: enhancement, material
---

Built-in materials live in `src/moire_metrology/materials.py`. Please open
this issue *before* writing the PR so we can discuss conventions and
parameters first. If you just want to use a new material in your own
scripts without upstreaming it, the README's "Custom materials" section
shows how to construct a `Material` directly.

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

## GSFE coefficients

Generalized stacking fault energy expansion `(c0, c1, c2, c3, c4, c5)` in
meV/uc.

- `c0`:
- `c1`:
- `c2`:
- `c3`:
- `c4`:
- `c5`:
- Citation:
- Basis convention (which Fourier basis the `c_i` are defined against):

## Stacking convention

H-stacking, R-stacking, homobilayer, heterobilayer — and any sign or
origin choices that matter for reproducing the GSFE landscape.

## Test case

A short end-to-end test (small relaxation, asserted energy or convergence
behaviour) that exercises the new material. Sketch what you'd assert.

## Additional notes

Anything else relevant — alternative parameter sets in the literature,
known caveats, temperature dependence, etc.
