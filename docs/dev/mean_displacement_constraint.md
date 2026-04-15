# Mean-displacement constraints

## Motivation

A finite-flake relaxation with free (Neumann) boundary conditions has
a 2D in-plane translation null mode: the energy is invariant under a
uniform rigid translation of the flake, so the Hessian is singular
and Newton-type solvers cannot run without some fix. The two existing
fixes available in the package have physical biases:

- **Pinning a single vertex** (`PinnedConstraints` on one vertex, or
  `pin_mean=True` in `build_outer_layer_constraints`) locks that
  vertex's stacking to an arbitrary point in the GSFE landscape —
  usually to AA, which is a GSFE *maximum*. The nearby relaxation is
  then pulled into a stacking pattern whose existence is an artefact
  of the pin location and amplitude, not of the flake geometry.
- **A soft spring tip** (`SpringTip` with small stiffness) lifts the
  null mode by a tiny amount but does not actively resist rigid-body
  translation — when the minimum-energy configuration involves a
  uniform lateral shift (e.g. after a `lateral_offset` change), the
  flake simply rigid-translates, and any intrinsic non-uniform
  response is hidden underneath the bulk translation.

`MeanDisplacementConstraint` (MDC) offers a third, bias-free option:
enforce

    (1/|S|) · Σ_{i ∈ S} u_{c,i} = target_c

for one or both components `c ∈ {x, y}` of a chosen vertex subset `S`
on one layer. With `target = (0, 0)` this holds the chosen layer's
centre of mass while leaving every individual vertex free. The null
mode is removed exactly, no vertex is singled out, and external
driving (for example via `lateral_offset` applied to the opposing
layer) is resisted by the internal degrees of freedom of the
constrained layer rather than leaking into a rigid translation.

Beyond null-mode removal, this is the idiomatic way to impose a
centre-of-mass boundary condition in any relaxation study —
tethered-layer calculations, imposed average strain on a flake, or
any setup that holds one layer's mean position fixed while others
relax.

## How it's enforced

The constraint is a linear equality `B · U = t` with `B` a
`(n_rows, n_sol)` sparse matrix and `t` a length-`n_rows` vector.
Inside `_newton_solve`, each Newton step solves the saddle-point
KKT system

    [  H + μ·I   B^T  ] [dU]   [  -g  ]
    [     B       0   ] [ λ ] = [ -c  ]

where `c = B U − t` is the current constraint residual and `λ` is
the Lagrange multiplier. The system is assembled via
`scipy.sparse.bmat` and solved with the existing sparse direct LU
(`spsolve`), which handles the indefinite saddle-point structure.
The same Levenberg-Marquardt damping `μ` logic applies to the upper
block `H + μ·I` as before.

If `U` starts on the constraint manifold (`c = 0`), each Newton step
remains on it: the KKT system enforces `B · dU = -c = 0`. With the
typical starting point `U = 0` and `target = (0, 0)`, the initial
residual is automatically zero.

If multiple MDCs are passed, `stack_mean_constraints` concatenates
their rows into a single `(m_total, n_free)` block. Overlap between
an MDC-touched DOF and a `PinnedConstraints` pinned DOF is rejected
at assembly time.

## Supported solver configurations

- `method="newton"` with `linear_solver="direct"` — full support.
- `method="newton"` with `linear_solver="iterative"` (CG) — not yet
  supported. CG requires a positive-definite operator; the KKT block
  is saddle-point (indefinite), so MINRES with a block preconditioner
  would be the principled extension.
- `method="pseudo_dynamics"` — not yet supported. The same
  saddle-point extension would apply to the per-step solve.
- `method="L-BFGS-B"` and other gradient-only scipy optimizers — not
  supported. For those, use a soft `SpringTip` at the flake centre as
  a practical approximation (at the cost of some localisation bias).

Attempting an unsupported configuration raises
`NotImplementedError` at `solve()` time.

## Example

```python
from moire_metrology import (
    GRAPHENE_GRAPHENE, MeanDisplacementConstraint,
    RelaxationSolver, SolverConfig, generate_finite_mesh,
)
from moire_metrology.discretization import Discretization, PinnedConstraints
from moire_metrology.lattice import HexagonalLattice, MoireGeometry

# Finite flake on a pinned substrate: fix all substrate DOFs.
lat = HexagonalLattice(alpha=GRAPHENE_GRAPHENE.bottom.lattice_constant)
geom = MoireGeometry(lat, theta_twist=1.0)
mesh = generate_finite_mesh(geom, n_cells=5, pixel_size=1.0)
disc = Discretization(mesh, geom)
conv = disc.build_conversion_matrices(nlayer1=1, nlayer2=1)

Nv = conv.n_vertices
pinned = {Nv + v for v in range(Nv)} | {3*Nv + v for v in range(Nv)}
pc = PinnedConstraints.from_indices(
    pinned_indices=sorted(pinned),
    pinned_values=[0.0] * len(pinned),
    n_full=conv.n_sol,
)

# Hold the top flake's centre of mass at the origin.
mdc = MeanDisplacementConstraint.from_layer(conv, layer_idx=0,
                                             target=(0.0, 0.0))

cfg = SolverConfig(method="newton", linear_solver="direct")
result = RelaxationSolver(cfg).solve(
    moire_interface=GRAPHENE_GRAPHENE,
    theta_twist=1.0,
    mesh=mesh,
    constraints=pc,
    mean_constraints=[mdc],
)
```

## Tests

`tests/test_mean_constraint.py`:

- `test_build_matrix_shape_and_target` — sparse matrix has the right
  shape, nonzero pattern is restricted to the right layer, and the
  target vector matches the constructor input.
- `test_stack_rejects_pinned_overlap` — mixing an MDC with a
  `PinnedConstraints` that pins one of the MDC-touched DOFs raises a
  clear error (the two constraints would be redundant or
  inconsistent).
- `test_mean_constraint_enforced_at_solution` — after an end-to-end
  Newton relaxation, `mean(u_x)` and `mean(u_y)` vanish to better
  than `10⁻⁸` for a `target = (0, 0)` MDC.
- `test_mean_constraint_differs_from_unconstrained_with_nonzero_target`
  — a nonzero target yields the exact mean as the result, and the
  relaxed energy differs from the `target = 0` case (so the
  constraint is active, not trivially satisfied).

Full regression: 136 / 136 pass (132 pre-existing + 4 new MDC).

## Limitations / future work

- Extending support to `linear_solver="iterative"` requires a
  block-preconditioned MINRES/GMRES for the saddle-point system.
- Extending support to `pseudo_dynamics` requires modifying its
  per-step linear solve to include the `B^T` augmentation (same
  pattern as the Newton path).
- The current API exposes a per-layer MDC; generalising to arbitrary
  linear constraints (e.g. `c_1 u_1 + c_2 u_2 = 0` coupling two
  layers) is straightforward but deferred until a concrete use case
  shows up.
