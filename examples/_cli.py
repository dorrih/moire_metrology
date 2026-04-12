"""Shared CLI helpers for the bundled examples.

This module is *not* part of the ``moire_metrology`` package — it lives
alongside the example scripts and provides a consistent interface for
selecting materials, configuring the solver, and handling errors
gracefully across all examples.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from moire_metrology import (
    BUNDLED_INTERFACES,
    Interface,
    SolverConfig,
)

# ── Bundled interface name registry ────────────────────────────────────

_INTERFACE_ALIASES: dict[str, Interface] = {}


def _norm(name: str) -> str:
    """Normalize a name for alias generation.

    Replaces ``'`` (prime) with ``p`` before stripping non-alnum chars,
    so ``hBN (AA')`` becomes ``hbn-aap`` rather than colliding with
    ``hBN (AA)`` → ``hbn-aa``.
    """
    return re.sub(
        r"[^a-z0-9]+", "-", name.lower().replace("'", "p").replace("\u2032", "p"),
    ).strip("-")


for _iface in BUNDLED_INTERFACES:
    _name_lower = _iface.name.lower().replace("'", "p").replace("\u2032", "p")
    # Exact lowered name (with prime → p).
    _INTERFACE_ALIASES[_iface.name.lower()] = _iface
    _INTERFACE_ALIASES[_name_lower] = _iface

    # Full slug: "MoSe2/WSe2 (H-stacked)" → "mose2-wse2-h-stacked"
    _slug = _norm(_iface.name)
    _INTERFACE_ALIASES[_slug] = _iface
    _INTERFACE_ALIASES[_slug.replace("-", "_")] = _iface

    # Short form: drop the second half of homobilayer names.
    # "graphene/graphene" → "graphene"
    if _iface.is_homobilayer:
        _short = _iface.name.lower().split("/")[0].strip()
        _short_slug = _norm(_short)
        _INTERFACE_ALIASES[_short] = _iface
        _INTERFACE_ALIASES[_short_slug] = _iface

    # For names with a parenthetical like "MoSe2/WSe2 (H-stacked)",
    # register without the parenthetical and with progressive prefixes
    # of the parenthetical content appended.
    if "(" in _iface.name:
        _paren_match = re.search(r"\(([^)]+)\)", _iface.name)
        _no_paren = re.sub(r"\s*\(.*\)", "", _iface.name).strip()
        _no_paren_slug = _norm(_no_paren)
        _INTERFACE_ALIASES[_no_paren.lower()] = _iface
        _INTERFACE_ALIASES[_no_paren_slug] = _iface
        _INTERFACE_ALIASES[_no_paren_slug.replace("-", "_")] = _iface
        if _paren_match:
            _paren_slug = _norm(_paren_match.group(1))
            _parts = _paren_slug.split("-")
            for _i in range(len(_parts)):
                _partial = "-".join(_parts[: _i + 1])
                _alias = f"{_no_paren_slug}-{_partial}"
                _INTERFACE_ALIASES[_alias] = _iface
                _INTERFACE_ALIASES[_alias.replace("-", "_")] = _iface

    # "hbn (aap) homobilayer" → "hbn-aap", "hbn_aap"
    if "homobilayer" in _iface.name.lower():
        _no_homo = re.sub(r"\s*homobilayer\s*", " ", _iface.name).strip()
        _no_homo_slug = _norm(_no_homo)
        _INTERFACE_ALIASES[_no_homo_slug] = _iface
        _INTERFACE_ALIASES[_no_homo_slug.replace("-", "_")] = _iface


def list_bundled_names() -> list[str]:
    """Return a deduplicated, sorted list of canonical interface names."""
    seen: set[int] = set()
    names: list[str] = []
    for _iface in BUNDLED_INTERFACES:
        if id(_iface) not in seen:
            seen.add(id(_iface))
            names.append(_iface.name)
    return names


def resolve_interface(name_or_path: str) -> Interface:
    """Look up a bundled interface by name, or load from a TOML file.

    Bundled names are matched case-insensitively with flexible
    separators (slash, space, underscore, hyphen).  If the string ends
    in ``.toml``, it is loaded via :meth:`Interface.from_toml`.

    On failure, prints available bundled names and exits with code 1
    (no raw traceback).
    """
    # TOML file path?
    if name_or_path.endswith(".toml"):
        path = Path(name_or_path)
        if not path.is_file():
            print(
                f"Error: TOML file not found: {path}\n\n"
                "Provide an absolute or relative path to a valid "
                "Interface TOML file.\n"
                "See docs/custom-materials.md for the schema reference.",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            return Interface.from_toml(path)
        except (ValueError, KeyError) as exc:
            print(f"Error loading interface from {path}:\n  {exc}", file=sys.stderr)
            sys.exit(1)

    # Bundled name lookup.
    key = name_or_path.lower().strip()
    iface = _INTERFACE_ALIASES.get(key)
    if iface is not None:
        return iface

    # Fuzzy: replace any non-alnum with dash, try again.
    key_slug = re.sub(r"[^a-z0-9]+", "-", key).strip("-")
    iface = _INTERFACE_ALIASES.get(key_slug)
    if iface is not None:
        return iface

    # Not found — helpful error.
    bundled = list_bundled_names()
    print(
        f"Error: unknown interface '{name_or_path}'.\n\n"
        "Bundled interfaces (use any of these names):\n"
        + "".join(f"  - {n}\n" for n in bundled)
        + "\nYou can also pass a path to a TOML file:\n"
        "  python example.py --interface path/to/my_interface.toml\n\n"
        "See docs/custom-materials.md for the TOML schema reference.",
        file=sys.stderr,
    )
    sys.exit(1)


def slugify(name: str) -> str:
    """Convert an interface name to a filesystem-safe slug.

    ``"MoSe2/WSe2 (H-stacked)"`` → ``"mose2_wse2_h-stacked"``
    """
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# ── Interface listing ──────────────────────────────────────────────────

def print_interface_list() -> None:
    """Print a detailed table of all bundled interfaces and exit."""
    print("Bundled interfaces")
    print("=" * 72)
    for iface in BUNDLED_INTERFACES:
        c = iface.gsfe_coeffs
        kind = "homobilayer" if iface.is_homobilayer else "heterointerface"
        print(f"\n  {iface.name}  ({kind})")
        print(f"    Bottom : {iface.bottom.name}  "
              f"a = {iface.bottom.lattice_constant:.4f} nm")
        print(f"    Top    : {iface.top.name}  "
              f"a = {iface.top.lattice_constant:.4f} nm")
        print(f"    GSFE   : c0={c[0]}, c1={c[1]}, c2={c[2]}, "
              f"c3={c[3]}, c4={c[4]}, c5={c[5]}  meV/uc")
        if iface.reference:
            print(f"    Ref    : {iface.reference}")

        # Show shortest useful alias.
        if iface.is_homobilayer and "/" in iface.name:
            short = iface.name.lower().split("/")[0].strip()
            cli_name = _norm(short)
        elif "homobilayer" in iface.name.lower():
            no_homo = re.sub(r"\s*homobilayer\s*", " ", iface.name).strip()
            cli_name = _norm(no_homo)
        elif "(" in iface.name:
            no_paren = re.sub(r"\s*\(.*\)", "", iface.name).strip()
            cli_name = _norm(no_paren)
        else:
            cli_name = _norm(iface.name)
        print(f"    CLI    : --interface {cli_name}")

    print("\n" + "-" * 72)
    print("Custom interfaces can be loaded from TOML files:")
    print("  --interface path/to/my_interface.toml")
    print("\nSee docs/custom-materials.md for the TOML schema reference.")
    sys.exit(0)


# ── argparse helpers ───────────────────────────────────────────────────

def add_interface_arg(
    parser: argparse.ArgumentParser,
    default: str,
) -> None:
    """Add ``--interface`` and ``--list-interfaces`` arguments."""
    bundled = ", ".join(list_bundled_names())
    parser.add_argument(
        "--interface",
        default=default,
        metavar="NAME_OR_TOML",
        help=(
            f"Bundled interface name or path to a TOML file. "
            f"Bundled: {bundled}. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--list-interfaces",
        action="store_true",
        help="List all bundled interfaces with parameters and exit.",
    )


def handle_list_interfaces(args: argparse.Namespace) -> None:
    """If ``--list-interfaces`` was passed, print the table and exit."""
    if getattr(args, "list_interfaces", False):
        print_interface_list()


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    default_interface: str,
    default_theta: float,
    default_pixel_size: float,
    default_method: str = "newton",
    default_max_iter: int = 200,
    default_gtol: float = 1e-6,
) -> None:
    """Add the standard set of CLI arguments shared across examples."""
    add_interface_arg(parser, default=default_interface)
    parser.add_argument(
        "--theta-twist",
        type=float,
        default=default_theta,
        metavar="DEG",
        help="Twist angle in degrees. Default: %(default)s",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=default_pixel_size,
        metavar="NM",
        help="Target mesh element size in nm. Default: %(default)s",
    )
    parser.add_argument(
        "--method",
        choices=["newton", "L-BFGS-B", "pseudo_dynamics"],
        default=default_method,
        help="Solver method. Default: %(default)s",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=default_max_iter,
        metavar="N",
        help="Maximum solver iterations. Default: %(default)s",
    )
    parser.add_argument(
        "--gtol",
        type=float,
        default=default_gtol,
        help="Absolute gradient norm tolerance. Default: %(default)s",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="Relative gradient tolerance (|grad|/|grad0|). Default: %(default)s",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plotting (headless / CI smoke test).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-solve even if a cached result exists.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output directory. Default: examples/output/",
    )


def build_solver_config(
    args: argparse.Namespace,
    **overrides: object,
) -> SolverConfig:
    """Build a :class:`SolverConfig` from parsed CLI args.

    Extra keyword arguments are forwarded to the constructor and
    override anything derived from *args*.
    """
    kwargs: dict[str, object] = dict(
        method=args.method,
        pixel_size=args.pixel_size,
        max_iter=args.max_iter,
        gtol=args.gtol,
        rtol=getattr(args, "rtol", 1e-4),
        display=True,
    )
    # Pass through iterative-solver args if present on the namespace.
    if getattr(args, "linear_solver", None) is not None:
        kwargs["linear_solver"] = args.linear_solver
    if getattr(args, "linear_solver_tol", None) is not None:
        kwargs["linear_solver_tol"] = args.linear_solver_tol
    if getattr(args, "linear_solver_maxiter", None) is not None:
        kwargs["linear_solver_maxiter"] = args.linear_solver_maxiter
    if getattr(args, "min_mesh_points", None) is not None:
        kwargs["min_mesh_points"] = args.min_mesh_points
    kwargs.update(overrides)
    return SolverConfig(**kwargs)


def get_output_dir(args: argparse.Namespace) -> Path:
    """Return the output directory, creating it if necessary."""
    if args.output_dir is not None:
        out = args.output_dir
    else:
        out = Path(__file__).resolve().parent / "output"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ── Validation helpers ─────────────────────────────────────────────────

def require_homobilayer(interface: Interface, example_name: str) -> None:
    """Exit gracefully if *interface* is not a homobilayer.

    Multi-layer examples need the same material on both sides of the
    moire interface to build a valid :class:`LayerStack`.
    """
    if not interface.is_homobilayer:
        print(
            f"Error: {example_name} requires a homobilayer interface "
            f"(same material on both sides of the moire interface).\n\n"
            f"The selected interface '{interface.name}' is a "
            f"heterointerface ({interface.bottom.name} / "
            f"{interface.top.name}).\n\n"
            "For heterointerface relaxation, use:\n"
            "  bilayer_relaxation.py",
            file=sys.stderr,
        )
        sys.exit(1)


def require_stacking_func(interface: Interface, example_name: str) -> None:
    """Exit gracefully if *interface* lacks a stacking function.

    TOML-loaded interfaces cannot carry a Python callable for the
    multi-layer stacking convention. Multi-layer examples that need
    ``stacking_func`` must use a bundled interface.
    """
    if interface.stacking_func is None:
        print(
            f"Error: {example_name} requires an interface with a "
            f"stacking function for the internal flake stacking "
            f"convention, but '{interface.name}' has stacking_func=None.\n\n"
            "TOML-loaded interfaces cannot carry a Python callable.\n"
            "Use one of the bundled homobilayer interfaces instead, or\n"
            "construct the Interface in Python with a custom stacking_func.\n\n"
            "Bundled homobilayer interfaces:\n"
            + "".join(
                f"  - {i.name}\n"
                for i in BUNDLED_INTERFACES
                if i.is_homobilayer
            ),
            file=sys.stderr,
        )
        sys.exit(1)


def print_interface_info(interface: Interface) -> None:
    """Print a summary of the interface and its materials."""
    print(f"Interface : {interface.name}")
    if interface.reference:
        print(f"Reference : {interface.reference}")
    _print_material("Bottom", interface.bottom)
    _print_material("Top   ", interface.top)
    c = interface.gsfe_coeffs
    print(f"GSFE (Carr, meV/uc): c0={c[0]}, c1={c[1]}, c2={c[2]}, "
          f"c3={c[3]}, c4={c[4]}, c5={c[5]}")
    if interface.is_homobilayer:
        print("Type      : homobilayer (centrosymmetric)"
              if c[4] == 0 and c[5] == 0
              else "Type      : homobilayer (non-centrosymmetric)")
    else:
        delta = abs(interface.top.lattice_constant
                    - interface.bottom.lattice_constant)
        avg = (interface.top.lattice_constant
               + interface.bottom.lattice_constant) / 2
        print(f"Type      : heterointerface "
              f"(lattice mismatch {delta / avg * 100:.2f}%)")
    print()


def _print_material(label: str, mat) -> None:
    k_npm, g_npm = mat.moduli_n_per_m
    print(f"  {label}: {mat.name}  "
          f"a={mat.lattice_constant:.4f} nm  "
          f"K={mat.bulk_modulus:.0f} meV/uc ({k_npm:.1f} N/m)  "
          f"G={mat.shear_modulus:.0f} meV/uc ({g_npm:.1f} N/m)")
