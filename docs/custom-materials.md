# Custom materials and interfaces

The package bundles five interfaces (see `--list-interfaces` on any
example script), but you can define your own materials and interfaces
from TOML files without forking the package.

## TOML schema

### Interface

An interface TOML file defines the moire interlayer coupling (GSFE)
and the two materials on either side.

```toml
[interface]
name = "MoSe2/WSe2 (H-stacked)"
gsfe_coeffs = [42.6, 16.0, -2.7, -1.1, 3.7, 0.6]
reference = "Shabani et al., Nat. Phys. 17, 720 (2021)"  # optional

[interface.bottom]
name = "WSe2"
lattice_constant = 0.3282    # nm
bulk_modulus = 43113.0       # meV/uc
shear_modulus = 30770.0      # meV/uc

[interface.top]
name = "MoSe2"
lattice_constant = 0.3288    # nm
bulk_modulus = 40521.0       # meV/uc
shear_modulus = 26464.0      # meV/uc
```

A complete worked example is at `examples/data/mose2_wse2_h.toml`.

### Fields

**Interface (required)**:

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Human-readable name |
| `gsfe_coeffs` | array of 6 floats | Carr-basis Fourier coefficients `[c0, c1, c2, c3, c4, c5]` in meV/uc |
| `bottom` | table | Bottom-layer material (see below) |
| `top` | table | Top-layer material (see below) |

**Interface (optional)**:

| Field | Type | Description |
|-------|------|-------------|
| `reference` | string | Literature citation |

**Material (required)**:

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Human-readable name |
| `lattice_constant` | float | In-plane lattice constant in nm |
| `bulk_modulus` | float | 2D bulk modulus K in meV/uc |
| `shear_modulus` | float | 2D shear modulus G in meV/uc |

## Unit conventions

All quantities are in the Carr/Halbertal convention:

- **Lattice constants**: nanometers (nm)
- **Elastic moduli** (K, G): meV per unit cell (meV/uc).
  The unit cell area is S_uc = (sqrt(3)/2) * a^2.
  To convert from the more common literature 2D moduli in N/m,
  use `Material.from_2d_moduli_n_per_m()` in Python, or multiply
  by `S_uc / (1.602e-22)` manually.
- **GSFE coefficients**: meV per unit cell, Carr basis.

### Carr GSFE Fourier expansion

```
V(v, w) = c0 + c1*(cos(v) + cos(w) + cos(v + w))
         + c2*(cos(v + 2w) + cos(v - w) + cos(2v + w))
         + c3*(cos(2v) + cos(2w) + cos(2v + 2w))
         + c4*(sin(v) + sin(w) - sin(v + w))
         + c5*(sin(2v + 2w) - sin(2v) - sin(2w))
```

For centrosymmetric homobilayers: `c4 = c5 = 0`.

For heterointerfaces with broken inversion symmetry (TMDs, hBN AA'):
`c4, c5` are generally non-zero.

## Finding GSFE coefficients in the literature

DFT-computed GSFE parameterizations for common 2D materials can be
found in:

- **Zhou et al.** PRB 92, 155438 (2015), Table III -- graphene, hBN
  (AA and AA'), graphene/hBN.  Uses a different Fourier basis than
  Carr; the package's `_zhou_to_carr()` transform handles conversion.
- **Carr et al.** PRB 98, 224102 (2018), Table I -- graphene (already
  in Carr basis).
- **Shabani, Halbertal et al.** Nat. Phys. 17, 720 (2021) -- TMD
  heterointerfaces (H-stacked MoSe2/WSe2, already in Carr basis).

When adding a new material from a paper, check:

1. Which Fourier basis the paper uses (Zhou vs Carr vs other).
2. Whether the coefficients are per unit cell (meV/uc) or per area
   (meV/nm^2).
3. The stacking convention (H vs R for TMDs, AA vs AA' for hBN).

## Limitations

- **`stacking_func` cannot be loaded from TOML.**  The stacking
  function (which defines how layers are offset in a multi-layer
  flake) is a Python callable and cannot be serialized to TOML.
  For bilayer examples this is irrelevant (the stacking function is
  only used for multi-layer stacks).  For multi-layer examples, you
  must use a bundled interface or construct the Interface in Python.

- **No automatic unit conversion in TOML.**  The TOML parser expects
  all values in the native meV/uc convention.  If your source paper
  reports moduli in N/m, convert before writing the TOML file.

## Error messages

The TOML loader produces clear error messages for common mistakes:

- **Missing fields**: "Material spec is missing required field(s): ..."
- **Extra fields** (e.g. putting GSFE on Material instead of Interface):
  "Material spec has unknown field(s): ... Did you mean to put
  GSFE-related fields on an Interface instead?"
- **Wrong GSFE length**: "gsfe_coeffs must have exactly 6 entries
  (c0..c5), got N"
- **Non-numeric GSFE**: "gsfe_coeffs must be a sequence of numbers"

## Using TOML files with the examples

All example scripts accept `--interface path/to/file.toml`:

```bash
python bilayer_relaxation.py --interface my_tmd.toml --theta-twist 1.0
```

If the TOML file is missing or malformed, the script exits with a
clear error (no raw traceback).
