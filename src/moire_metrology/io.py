"""I/O utilities for saving, loading, and importing results."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_matlab_result(path: str | Path) -> dict:
    """Load a MATLAB .mat result file.

    Parameters
    ----------
    path : str or Path
        Path to the .mat file.

    Returns
    -------
    dict
        Dictionary of arrays from the MATLAB file.
    """
    from scipy.io import loadmat

    data = loadmat(str(path), squeeze_me=True)
    # Remove MATLAB metadata keys
    return {k: v for k, v in data.items() if not k.startswith("__")}
