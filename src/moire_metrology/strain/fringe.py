"""Moire fringe data loading and interpolation.

Handles loading traced moire fringe data from various formats
and converting to the registry field representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import UnivariateSpline

from .polynomial import RegistryField


@dataclass
class FringeLine:
    """A single traced moire fringe line.

    Attributes
    ----------
    x : ndarray
        x-coordinates along the fringe (nm).
    y : ndarray
        y-coordinates along the fringe (nm).
    index : int
        Integer registry index assigned to this fringe (I or J value).
    family : int
        Fringe family: 1 for I-fringes, 2 for J-fringes.
    """

    x: np.ndarray
    y: np.ndarray
    index: int
    family: int


@dataclass
class FringeSet:
    """Collection of traced moire fringes for strain extraction.

    Attributes
    ----------
    fringes : list of FringeLine
        All traced fringe lines.
    """

    fringes: list[FringeLine] = field(default_factory=list)

    @property
    def i_fringes(self) -> list[FringeLine]:
        """Fringes of family 1 (I-type)."""
        return [f for f in self.fringes if f.family == 1]

    @property
    def j_fringes(self) -> list[FringeLine]:
        """Fringes of family 2 (J-type)."""
        return [f for f in self.fringes if f.family == 2]

    def fit_registry_fields(
        self,
        order: int = 8,
        resample_density: float = 5.0,
    ) -> tuple[RegistryField, RegistryField]:
        """Fit polynomial registry fields I(x,y) and J(x,y) to the fringe data.

        Parameters
        ----------
        order : int
            Polynomial order for the 2D fit (default 8).
        resample_density : float
            Points per nm when resampling fringe curves (default 5.0).

        Returns
        -------
        I_field, J_field : RegistryField
            Polynomial fits to the I and J registry fields.
        """
        # Collect all I-fringe and J-fringe data points
        x_I, y_I, val_I = _collect_fringe_points(self.i_fringes, resample_density)
        x_J, y_J, val_J = _collect_fringe_points(self.j_fringes, resample_density)

        I_field = RegistryField.fit(x_I, y_I, val_I, order=order)
        J_field = RegistryField.fit(x_J, y_J, val_J, order=order)

        return I_field, J_field

    def estimate_moire_wavelength(self) -> float:
        """Estimate the moire wavelength from fringe spacing."""
        lambdas = []
        for family_fringes in [self.i_fringes, self.j_fringes]:
            if len(family_fringes) < 2:
                continue
            indices = sorted(set(f.index for f in family_fringes))
            if len(indices) < 2:
                continue
            # Average spacing between first and last fringe
            first = [f for f in family_fringes if f.index == indices[0]][0]
            last = [f for f in family_fringes if f.index == indices[-1]][0]
            cx1, cy1 = np.mean(first.x), np.mean(first.y)
            cx2, cy2 = np.mean(last.x), np.mean(last.y)
            dist = np.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)
            n_spacings = indices[-1] - indices[0]
            if n_spacings > 0:
                lambdas.append(dist / n_spacings)

        if not lambdas:
            raise ValueError("Not enough fringes to estimate wavelength")
        return np.sqrt(np.prod(lambdas)) if len(lambdas) == 2 else lambdas[0]

    @classmethod
    def from_csv(cls, path: str | Path) -> FringeSet:
        """Load fringes from a CSV file.

        Expected format: x, y, index, family
        Each row is a point; fringes are separated by blank lines or by
        changes in the (index, family) pair.
        """
        path = Path(path)
        data = np.loadtxt(path, delimiter=",", comments="#")

        fringes = []
        if data.shape[1] >= 4:
            # Group by (index, family)
            unique_keys = set(map(tuple, data[:, 2:4].astype(int)))
            for idx, fam in unique_keys:
                mask = (data[:, 2].astype(int) == idx) & (data[:, 3].astype(int) == fam)
                pts = data[mask]
                fringes.append(FringeLine(
                    x=pts[:, 0], y=pts[:, 1], index=int(idx), family=int(fam),
                ))

        return cls(fringes=fringes)

    @classmethod
    def from_matlab(cls, path: str | Path) -> FringeSet:
        """Load fringes from a MATLAB .mat file (GUI output format).

        Expects keys: xpts_list, ypts_list, line_integer_val, line_type_list.
        """
        from scipy.io import loadmat

        data = loadmat(str(path), squeeze_me=True)
        xpts = data["xpts_list"]
        ypts = data["ypts_list"]
        indices = data["line_integer_val"]
        families = data["line_type_list"]

        fringes = []
        for k in range(len(xpts)):
            x = np.asarray(xpts[k], dtype=float).ravel()
            y = np.asarray(ypts[k], dtype=float).ravel()
            fringes.append(FringeLine(
                x=x, y=y,
                index=int(indices[k]),
                family=int(families[k]),
            ))

        return cls(fringes=fringes)


def _collect_fringe_points(
    fringes: list[FringeLine],
    resample_density: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample and collect all points from a list of fringes.

    Each fringe is resampled with cubic spline interpolation to ensure
    uniform dense sampling along the curve.
    """
    all_x, all_y, all_val = [], [], []

    for fringe in fringes:
        if len(fringe.x) < 2:
            continue

        # Compute cumulative arc length
        dx = np.diff(fringe.x)
        dy = np.diff(fringe.y)
        ds = np.sqrt(dx**2 + dy**2)
        s = np.concatenate([[0], np.cumsum(ds)])
        total_length = s[-1]

        if total_length < 1e-10:
            continue

        # Resample with spline interpolation
        n_resample = max(int(total_length * resample_density), 2)
        s_new = np.linspace(0, total_length, n_resample)

        try:
            spline_x = UnivariateSpline(s, fringe.x, k=min(3, len(s) - 1), s=0)
            spline_y = UnivariateSpline(s, fringe.y, k=min(3, len(s) - 1), s=0)
            x_new = spline_x(s_new)
            y_new = spline_y(s_new)
        except Exception:
            x_new = np.interp(s_new, s, fringe.x)
            y_new = np.interp(s_new, s, fringe.y)

        all_x.append(x_new)
        all_y.append(y_new)
        all_val.append(np.full(n_resample, fringe.index, dtype=float))

    if not all_x:
        return np.array([]), np.array([]), np.array([])

    return np.concatenate(all_x), np.concatenate(all_y), np.concatenate(all_val)
