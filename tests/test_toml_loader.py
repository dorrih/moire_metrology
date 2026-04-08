"""Tests for the Material.from_toml / Interface.from_toml loaders."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from moire_metrology import (
    Interface,
    Material,
    MOSE2_WSE2_H_INTERFACE,
    RelaxationSolver,
    SolverConfig,
)


# ---------------------------------------------------------------------------
# Material.from_dict / Material.from_toml
# ---------------------------------------------------------------------------


class TestMaterialFromDict:
    def test_round_trip_basic(self):
        m = Material.from_dict({
            "name": "MyTMD",
            "lattice_constant": 0.330,
            "bulk_modulus": 42000.0,
            "shear_modulus": 28000.0,
        })
        assert m.name == "MyTMD"
        assert m.lattice_constant == 0.330
        assert m.bulk_modulus == 42000.0
        assert m.shear_modulus == 28000.0

    def test_missing_field_raises_with_clear_message(self):
        with pytest.raises(ValueError, match="missing required field"):
            Material.from_dict({"name": "X", "lattice_constant": 0.3})

    def test_extra_field_raises_with_helpful_pointer(self):
        # Common mistake: putting GSFE on Material instead of Interface.
        with pytest.raises(ValueError, match="unknown field"):
            Material.from_dict({
                "name": "X", "lattice_constant": 0.3,
                "bulk_modulus": 1.0, "shear_modulus": 1.0,
                "gsfe_coeffs": [1, 2, 3, 4, 5, 6],
            })

    def test_int_coerces_to_float(self):
        # Users may write `bulk_modulus = 8595` in TOML expecting it to work.
        m = Material.from_dict({
            "name": "X", "lattice_constant": 0.247,
            "bulk_modulus": 8595, "shear_modulus": 5765,
        })
        assert isinstance(m.bulk_modulus, float)
        assert m.bulk_modulus == 8595.0


class TestMaterialFromToml:
    def test_round_trip_from_file(self, tmp_path: Path):
        path = tmp_path / "mytmd.toml"
        path.write_text(textwrap.dedent("""\
            [material]
            name = "MyTMD"
            lattice_constant = 0.330
            bulk_modulus = 42000.0
            shear_modulus = 28000.0
        """))
        m = Material.from_toml(path)
        assert m == Material(
            name="MyTMD",
            lattice_constant=0.330,
            bulk_modulus=42000.0,
            shear_modulus=28000.0,
        )

    def test_missing_top_level_table_raises(self, tmp_path: Path):
        path = tmp_path / "no_table.toml"
        path.write_text("name = \"X\"\n")
        with pytest.raises(ValueError, match=r"\[material\] table"):
            Material.from_toml(path)


# ---------------------------------------------------------------------------
# Interface.from_dict / Interface.from_toml
# ---------------------------------------------------------------------------


_VALID_INTERFACE_DICT = {
    "name": "MoSe2/WSe2 (H-stacked)",
    "gsfe_coeffs": [42.6, 16.0, -2.7, -1.1, 3.7, 0.6],
    "reference": "Shabani et al., Nat. Phys. 17, 720 (2021)",
    "bottom": {
        "name": "WSe2",
        "lattice_constant": 0.3282,
        "bulk_modulus": 43113.0,
        "shear_modulus": 30770.0,
    },
    "top": {
        "name": "MoSe2",
        "lattice_constant": 0.3288,
        "bulk_modulus": 40521.0,
        "shear_modulus": 26464.0,
    },
}


class TestInterfaceFromDict:
    def test_round_trip_basic(self):
        i = Interface.from_dict(_VALID_INTERFACE_DICT)
        assert i.name == "MoSe2/WSe2 (H-stacked)"
        assert i.bottom.name == "WSe2"
        assert i.top.name == "MoSe2"
        assert i.gsfe_coeffs == (42.6, 16.0, -2.7, -1.1, 3.7, 0.6)
        assert i.reference == "Shabani et al., Nat. Phys. 17, 720 (2021)"
        assert i.stacking_func is None
        assert not i.is_homobilayer

    def test_homobilayer_via_two_identical_inline_materials(self):
        spec = {
            "name": "Graphene homo (custom)",
            "gsfe_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "bottom": {
                "name": "Graphene", "lattice_constant": 0.247,
                "bulk_modulus": 8595.0, "shear_modulus": 5765.0,
            },
            "top": {
                "name": "Graphene", "lattice_constant": 0.247,
                "bulk_modulus": 8595.0, "shear_modulus": 5765.0,
            },
        }
        i = Interface.from_dict(spec)
        # Two distinct Material instances with the same content — they
        # are *equal* (frozen dataclass equality) but not the *same*
        # object, so is_homobilayer correctly returns False here. The
        # is_homobilayer property is for the bundled-constant case
        # where both sides are literally the same Python object.
        assert i.bottom == i.top
        assert i.bottom is not i.top

    def test_reference_optional(self):
        spec = dict(_VALID_INTERFACE_DICT)
        del spec["reference"]
        i = Interface.from_dict(spec)
        assert i.reference == ""

    def test_missing_required_field_raises(self):
        spec = dict(_VALID_INTERFACE_DICT)
        del spec["gsfe_coeffs"]
        with pytest.raises(ValueError, match="missing required field"):
            Interface.from_dict(spec)

    def test_extra_field_raises_with_stacking_func_hint(self):
        spec = dict(_VALID_INTERFACE_DICT)
        spec["stacking_func"] = "graphene_bernal"  # not loadable
        with pytest.raises(ValueError, match="stacking_func is not loadable"):
            Interface.from_dict(spec)

    def test_wrong_gsfe_coeffs_length_raises(self):
        spec = dict(_VALID_INTERFACE_DICT)
        spec["gsfe_coeffs"] = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="exactly 6 entries"):
            Interface.from_dict(spec)

    def test_non_numeric_gsfe_coeffs_raises(self):
        spec = dict(_VALID_INTERFACE_DICT)
        spec["gsfe_coeffs"] = [1.0, "two", 3.0, 4.0, 5.0, 6.0]
        with pytest.raises(ValueError, match="sequence of numbers"):
            Interface.from_dict(spec)


class TestInterfaceFromToml:
    def test_round_trip_from_file(self, tmp_path: Path):
        path = tmp_path / "mose2_wse2.toml"
        path.write_text(textwrap.dedent("""\
            [interface]
            name = "MoSe2/WSe2 (H-stacked)"
            gsfe_coeffs = [42.6, 16.0, -2.7, -1.1, 3.7, 0.6]
            reference = "Shabani et al., Nat. Phys. 17, 720 (2021)"

            [interface.bottom]
            name = "WSe2"
            lattice_constant = 0.3282
            bulk_modulus = 43113.0
            shear_modulus = 30770.0

            [interface.top]
            name = "MoSe2"
            lattice_constant = 0.3288
            bulk_modulus = 40521.0
            shear_modulus = 26464.0
        """))
        i = Interface.from_toml(path)
        assert i.name == "MoSe2/WSe2 (H-stacked)"
        assert i.bottom.name == "WSe2"
        assert i.top.name == "MoSe2"
        assert i.gsfe_coeffs == (42.6, 16.0, -2.7, -1.1, 3.7, 0.6)

    def test_missing_top_level_table_raises(self, tmp_path: Path):
        path = tmp_path / "no_table.toml"
        path.write_text("name = \"X\"\n")
        with pytest.raises(ValueError, match=r"\[interface\] table"):
            Interface.from_toml(path)

    def test_loaded_interface_matches_bundled_end_to_end(self, tmp_path: Path):
        """A TOML-loaded interface with the same coefficients as a
        bundled one must produce *bit-identical* relaxation results
        — anything else means the loader silently dropped or
        transformed a field."""
        path = tmp_path / "mose2_wse2.toml"
        path.write_text(textwrap.dedent("""\
            [interface]
            name = "MoSe2/WSe2 (H-stacked)"
            gsfe_coeffs = [42.6, 16.0, -2.7, -1.1, 3.7, 0.6]
            reference = "Shabani et al., Nat. Phys. 17, 720 (2021)"

            [interface.bottom]
            name = "WSe2"
            lattice_constant = 0.3282
            bulk_modulus = 43113.0
            shear_modulus = 30770.0

            [interface.top]
            name = "MoSe2"
            lattice_constant = 0.3288
            bulk_modulus = 40521.0
            shear_modulus = 26464.0
        """))
        loaded = Interface.from_toml(path)

        cfg = SolverConfig(
            method="L-BFGS-B", pixel_size=1.5, max_iter=80,
            gtol=1e-3, display=False, min_mesh_points=30,
        )
        bundled = RelaxationSolver(cfg).solve(
            moire_interface=MOSE2_WSE2_H_INTERFACE, theta_twist=2.0,
        )
        user = RelaxationSolver(cfg).solve(
            moire_interface=loaded, theta_twist=2.0,
        )
        assert user.total_energy == bundled.total_energy, (
            f"TOML-loaded interface produced different energy than bundled "
            f"({user.total_energy} vs {bundled.total_energy}); the loader "
            f"silently dropped or transformed a GSFE coefficient."
        )
