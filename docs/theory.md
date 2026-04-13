# Theory and background

This page summarises the continuum model behind the package. For full
derivations see the papers listed on the {doc}`references` page.

## Moire superlattice geometry

When two hexagonal crystals with lattice constants $\alpha_1$ and
$\alpha_2$ are stacked with a relative twist angle $\theta$, their
real-space interference produces a moire superlattice with lattice
vectors $\mathbf{V}_1$ and $\mathbf{V}_2$. The moire wavelength scales
as $\lambda \sim \alpha / \theta$ for small twist (homobilayer case), or
is set by the lattice mismatch for a heterointerface at zero twist
(e.g. graphene on hBN, $\lambda \approx 15.75\,\text{nm}$ from the
1.6% mismatch).

The package computes the moire geometry from the input lattice constants
and twist angle, and tiles the moire unit cell with a triangular FEM
mesh.

## Energy functional

The solver minimises the total energy per moire unit cell,

$$
E = E_{\text{elastic}} + E_{\text{GSFE}},
$$

where the two terms model intra-layer strain and inter-layer stacking,
respectively.

### Elastic energy

Each layer is treated as a 2D isotropic elastic sheet. The elastic
energy density per unit cell is

$$
E_{\text{elastic}} = \tfrac{1}{2}\,K\,(\partial_x u_x + \partial_y u_y)^2
    + \tfrac{1}{2}\,G\,\bigl[(\partial_x u_x - \partial_y u_y)^2
    + (\partial_x u_y + \partial_y u_x)^2\bigr],
$$

where $\mathbf{u}(\mathbf{r})$ is the in-plane displacement field,
$K = \lambda + \mu$ is the 2D bulk modulus, and $G = \mu$ is the 2D
shear modulus. Both are stored in **meV per unit cell**, the
Carr/Halbertal convention used throughout the package. See the
{doc}`custom materials page <custom-materials>` for unit conversion
details.

The elastic Hessian is constant (depends only on the mesh and moduli),
so it is assembled once as a sparse matrix and reused across iterations.

### Generalised Stacking Fault Energy (GSFE)

The GSFE captures the registry-dependent interlayer interaction. It is
parameterised as a Fourier expansion in stacking phase coordinates
$(v, w)$:

$$
V(v, w) = c_0 + c_1\,(\cos v + \cos w + \cos(v + w))
    + c_2\,(\cos(v + 2w) + \cos(v - w) + \cos(2v + w))
    + c_3\,(\cos 2v + \cos 2w + \cos(2v + 2w))
$$
$$
    + c_4\,(\sin v + \sin w - \sin(v + w))
    + c_5\,(\sin(2v + 2w) - \sin 2v - \sin 2w)
$$

The coefficients $(c_0, \ldots, c_5)$ are in meV per unit cell and
follow the Carr et al. basis convention.

- For **centrosymmetric homobilayers** (graphene/graphene, hBN-AA):
  $c_4 = c_5 = 0$.
- For **heterointerfaces with broken inversion symmetry** (MoSe2/WSe2 H,
  hBN-AA'): $c_4, c_5 \neq 0$.

The stacking phases $(v, w)$ depend on the local displacement difference
between the two layers and on the position-dependent moire phase.
Because $V$ is a smooth trigonometric surface, its gradient and Hessian
are computed analytically — no finite differences needed.

### Material vs interface

GSFE is a property of the *pair* of materials in contact, not of either
material individually. The package separates this cleanly:

- `Material` — per-layer: lattice constant, bulk modulus, shear modulus.
- `Interface` — per-pair: GSFE coefficients, stacking convention,
  literature reference.

## Solvers

The package provides three optimisation strategies for the energy
minimisation:

**Newton** (default for bilayer):
  Modified Newton method with analytical elastic + GSFE Hessians.
  Negative eigenvalues of the per-vertex 2x2 GSFE Hessian blocks are
  flipped to ensure positive-definiteness, enabling meaningful Newton
  steps from iteration 1. Assembled as a sparse system and solved with
  a direct factorisation. Typically converges in 10--30 iterations for
  bilayer problems.

**pseudo_dynamics** (recommended for multi-layer / stiff problems):
  Implicit theta-method time-stepping on the gradient flow, ported from
  the paper's MATLAB code. Unconditionally stable for $\theta \geq 0.5$.
  Has an opt-in matrix-free MINRES path for large meshes where direct
  factorisation is too expensive.

**L-BFGS-B** (fallback):
  SciPy's limited-memory BFGS with box constraints. No Hessian needed.
  Slower to converge but robust. Useful for quick sanity checks.

### Convergence criteria

Three criteria, any of which can trigger convergence:

- **gtol** — absolute gradient norm: $\|\nabla E\| < \text{gtol}$.
- **rtol** — relative gradient norm: $\|\nabla E\| / \|\nabla E_0\| < \text{rtol}$.
  Important at low twist angles where the absolute gradient at the
  minimum can be O(1--10) due to large total energies.
- **etol** — energy stagnation: fractional energy improvement over the
  last `etol_window` steps falls below `etol`.

## Multi-layer stacks

For $N$-layer flakes, the `LayerStack` API assembles GSFE coupling at:

- The **twisted interface** between the two flakes (always present).
- **Intra-stack** adjacent-layer pairs within each flake (when
  $n > 1$), using the homobilayer interface with the appropriate Bernal
  stacking offset.

Optional `fix_top` / `fix_bottom` flags clamp the outermost layers to
$\mathbf{u} = 0$, approximating semi-infinite substrates.

## Strain extraction

Given observed moire lattice vectors
$(\lambda_1, \lambda_2, \varphi_1, \varphi_2)$, the package recovers
the twist angle $\theta$ and the strain tensor components $(\varepsilon_c, \varepsilon_s)$ via
the closed-form inversion of Halbertal et al., ACS Nano (2022),
Eq. 9. Both pointwise and spatially-varying (polynomial registry field)
workflows are supported.
