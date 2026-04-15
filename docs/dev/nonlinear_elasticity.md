# Geometrically nonlinear elastic term

## Motivation

The current elastic energy uses linearized (Cauchy) strain
`ε = ½(∇u + ∇u^T)`. Cauchy strain is not rotation-invariant at finite
rotation: a rigid body rotation by angle α produces a symmetric
strain `(cos α − 1) · I` and therefore a spurious isotropic strain
energy of order `K · α⁴` per unit area. For small-twist relaxation
studies (|∇u| well below one percent) this inflation is below the
noise floor. For problems where one flake undergoes, or approaches,
finite rotation relative to the other — flake reorientation,
substrate-coupled rotations, large-displacement relaxation near
defects or pins, or relaxed states whose `u` fields have
rotation-like gradient content comparable to their strain-like
content — the Cauchy term is no longer a neutral modeling choice and
can bias the ground state selected by the solver.

The principled fix is to replace Cauchy strain with an objective
(rotation-invariant) strain measure. This document is the design
sketch for that change.

## What to change

Replace Cauchy strain with an objective strain measure. The two
standard options:

1. **Green–Lagrange strain.** Deformation gradient `F = I + ∇u`,
   strain `E = ½(F^T F − I)`. Vanishes identically for `F = R(α)` at
   any α. Energy density becomes quartic in `∇u` rather than
   quadratic, so the elastic Hessian is no longer constant — it must
   be reassembled each iteration like the GSFE Hessian.
2. **Polar-decomposition-based strain.** Write `F = R · U`, use
   `ln U` (or `U − I`) as the strain. Also rotation-invariant, but
   the decomposition at every triangle is more expensive than
   assembling Green–Lagrange directly.

Green–Lagrange is the conventional choice for continuum FE codes and
is the right path here. The corresponding constitutive law is the
St.-Venant–Kirchhoff model, which reuses the current `(K, G)` (or
Lamé `(λ, μ)`) material parameters unchanged.

## Implementation sketch

- Rename the current constant-Hessian elastic path and keep it
  available behind a flag:
  `SolverConfig.elastic_strain = "cauchy" | "green_lagrange"`,
  defaulting to `"cauchy"` until the nonlinear path is validated
  and benchmarked on the existing test corpus.
- New assembly path for `"green_lagrange"`:
  - At each triangle, evaluate `F = I + ∇u` from the current `U`
    using the existing linear-triangle shape-function gradients
    `dN/dx, dN/dy` built in `Discretization`.
  - Compute `E = ½(F^T F − I)` and the second Piola–Kirchhoff
    stress `S = λ tr(E) I + 2μ E`.
  - Energy density `ψ = ½ λ tr(E)² + μ E : E`; multiply by triangle
    area for the contribution to the total energy.
  - First variation (gradient): `∂ψ/∂F = F · S`. Scatter through
    `dN/dx, dN/dy` to each triangle's three vertex DOFs, exactly
    the same pattern the GSFE term already uses.
  - Second variation (Hessian): material tangent
    `C_ijkl = λ δ_ij δ_kl + μ (δ_ik δ_jl + δ_il δ_jk)` pushed through
    `F` gives one contribution; a geometric stiffness term
    proportional to `S` gives the other. Both reuse the sparsity
    pattern that the constant Cauchy Hessian already populates, so
    the solver's linear-algebra path is unchanged.
- The existing PinnedConstraints, multi-layer conversion matrices,
  and modified-eigenvalue Hessian projection all carry over
  unmodified — they operate on the full DOF vector and don't care
  how the elastic block was assembled.

## Tests to add

- **Rigid rotation returns zero.** For `u(r) = [R(α) − I] · (r − r₀)`
  with α up to a few degrees, assert the Green–Lagrange elastic
  energy is zero to numerical precision, while the Cauchy energy is
  nonzero (regression-protecting the difference).
- **Small-strain consistency.** For `|∇u| ≪ 1`, the linearized and
  nonlinear energies must agree to `O(|∇u|²)`; the difference on a
  known small-strain test field should be `O(|∇u|³)`.
- **Gradient and `hessp` vs finite differences.** Same pattern as
  the existing `tests/` finite-difference checks. The Hessian is no
  longer constant, so this is the main correctness check during
  development.
- **Regression against the existing test corpus.** With
  `elastic_strain = "cauchy"` (the default), every existing test
  must pass unchanged. With `elastic_strain = "green_lagrange"`,
  small-twist tests (e.g. the bundled θ = 2° relaxation fixtures)
  should agree with the Cauchy result to within a few percent.
- **Large-rotation reference.** For a bilayer that should relax to a
  globally rotated state (e.g. a single flake on a rigid substrate
  with a rigid-body rotation degree of freedom) the Green–Lagrange
  path should find the rotated minimum without any spurious elastic
  penalty.

## Performance considerations

- Per-iteration elastic-Hessian reassembly is the main new cost.
  Sparsity structure is identical to the current constant Hessian,
  so the pattern can be precomputed once; only the data array is
  refreshed per iteration.
- For the paper-sized problems in the existing test corpus
  (O(10⁴) DOFs, Newton), this should still be small relative to the
  sparse linear solve per Newton step.
- For very large meshes where the Cauchy constant-Hessian precompute
  is genuinely amortizing across many iterations, the default
  should remain Cauchy unless the user explicitly opts in.

## When to merge

After gradient/Hessian finite-difference tests pass, the small-strain
limit agrees with Cauchy within a few percent on the bundled tests,
and at least one rigid-rotation regression test is green. The default
stays `"cauchy"` on initial merge; switching the default is a
separate, later decision once the nonlinear path has been exercised
across the full test matrix.
