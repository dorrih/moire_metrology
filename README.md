# moire-metrology

Atomic relaxation and strain extraction for twisted 2D van der Waals heterostructures.

## Installation

```bash
pip install moire-metrology
```

## Quick Start

```python
from moire_metrology import RelaxationSolver, GRAPHENE

solver = RelaxationSolver()
result = solver.solve(
    material1=GRAPHENE,
    material2=GRAPHENE,
    theta_twist=2.0,  # degrees
)
result.plot_stacking()
```

## References

- Halbertal et al., "Moiré metrology of energy landscapes in van der Waals heterostructures", [arXiv:2008.04835](https://arxiv.org/abs/2008.04835)
- Shabani, Halbertal et al., "Deep moiré potentials in twisted transition metal dichalcogenide bilayers", [arXiv:2008.07696](https://arxiv.org/abs/2008.07696)
- Halbertal et al., "Multi-layered atomic relaxation in van der Waals heterostructures", [arXiv:2206.06395](https://arxiv.org/abs/2206.06395)
