# Quick start

## Bilayer relaxation

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

For a complete worked example reproducing the hallmark TBG relaxation
pattern, see `examples/bilayer_relaxation.py`. The example caches the
relaxed state to a `.npz` so you can iterate on plots without re-solving.

All examples are CLI-configurable:

```bash
# Default: twisted bilayer graphene at 0.2 deg
python examples/bilayer_relaxation.py

# Graphene on hBN (pure lattice-mismatch moire)
python examples/bilayer_relaxation.py --preset hbn

# H-stacked MoSe2/WSe2 (deep moire potential)
python examples/bilayer_relaxation.py --preset tmd

# Custom interface from a TOML file
python examples/bilayer_relaxation.py --interface my_interface.toml --theta-twist 0.5

# List all bundled interfaces
python examples/bilayer_relaxation.py --list-interfaces
```

See {doc}`examples` for the full guide to bundled examples, and
{doc}`custom-materials` for the TOML schema reference.

## Strain extraction from a measured moire

```python
from moire_metrology.strain import get_strain_minimize_compression

# Observed moire vectors (lengths in nm, angles in degrees)
result = get_strain_minimize_compression(
    alpha1=0.247, alpha2=0.247,
    lambda1=10.0, lambda2=10.0,
    phi1_deg=0.0, phi2_deg=60.0,
)
print(f"twist:           {result.theta_twist:.4f} deg")
print(f"compression e_c: {result.eps_c:.2e}")
print(f"shear e_s:       {result.eps_s:.2e}")
```

## Multi-layer stack

```python
from moire_metrology import GRAPHENE_GRAPHENE, SolverConfig
from moire_metrology.multilayer import LayerStack

stack = LayerStack(
    moire_interface=GRAPHENE_GRAPHENE,
    bottom_interface=GRAPHENE_GRAPHENE,  # required because n_bottom > 1
    n_top=1, n_bottom=2,
    theta_twist=1.5,
)
result = stack.solve(SolverConfig(method="L-BFGS-B", pixel_size=1.0))
```

## Constrained relaxation on a finite domain

```python
from moire_metrology import GRAPHENE, GRAPHENE_GRAPHENE, RelaxationSolver, SolverConfig
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
    moire_interface=GRAPHENE_GRAPHENE, theta_twist=2.0,
    mesh=mesh, constraints=constraints,
)
```
