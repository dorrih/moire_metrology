"""Visualization utilities for moire relaxation results.

Uses matplotlib's Triangulation and tripcolor for efficient rendering
of scalar fields on triangular meshes, with optional tiling to show
multiple moire unit cells.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


def plot_scalar_field(
    mesh,
    values: np.ndarray,
    ax=None,
    n_tile: int = 1,
    title: str = "",
    cmap: str = "viridis",
    colorbar: bool = True,
    **kwargs,
) -> plt.Axes:
    """Plot a scalar field on the moire mesh.

    Parameters
    ----------
    mesh : MoireMesh
        The triangular mesh.
    values : ndarray, shape (Nv,)
        Scalar values at each vertex.
    ax : matplotlib Axes or None
        If None, creates a new figure.
    n_tile : int
        Number of periodic copies in each direction (for visualization).
    title : str
        Plot title.
    cmap : str
        Colormap name.
    colorbar : bool
        Whether to add a colorbar.
    **kwargs
        Passed to tripcolor.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    else:
        fig = ax.figure

    V1 = mesh.V1
    V2 = mesh.V2
    Nv = mesh.n_vertices
    Nt = mesh.n_triangles

    # Build tiled mesh
    all_x = []
    all_y = []
    all_tri = []
    all_vals = []

    for i in range(n_tile):
        for j in range(n_tile):
            offset_x = i * V1[0] + j * V2[0]
            offset_y = i * V1[1] + j * V2[1]
            vert_offset = (i * n_tile + j) * Nv

            all_x.append(mesh.points[0] + offset_x)
            all_y.append(mesh.points[1] + offset_y)
            all_tri.append(mesh.triangles + vert_offset)
            all_vals.append(values)

    x_all = np.concatenate(all_x)
    y_all = np.concatenate(all_y)
    tri_all = np.concatenate(all_tri, axis=0)
    vals_all = np.concatenate(all_vals)

    tri_obj = Triangulation(x_all, y_all, tri_all)

    tc = ax.tripcolor(tri_obj, vals_all, cmap=cmap, shading="gouraud", **kwargs)
    if colorbar:
        fig.colorbar(tc, ax=ax)

    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    if title:
        ax.set_title(title)

    return ax


def plot_displacement_field(
    mesh,
    ux: np.ndarray,
    uy: np.ndarray,
    ax=None,
    n_tile: int = 1,
    scale: float = 1.0,
    **kwargs,
) -> plt.Axes:
    """Plot displacement vectors on the mesh.

    Parameters
    ----------
    mesh : MoireMesh
        The triangular mesh.
    ux, uy : ndarray, shape (Nv,)
        Displacement components.
    ax : matplotlib Axes or None
    n_tile : int
        Periodic tiling.
    scale : float
        Arrow scale factor.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    V1 = mesh.V1
    V2 = mesh.V2

    for i in range(n_tile):
        for j in range(n_tile):
            offset_x = i * V1[0] + j * V2[0]
            offset_y = i * V1[1] + j * V2[1]

            x = mesh.points[0] + offset_x
            y = mesh.points[1] + offset_y

            # Subsample for readability
            step = max(1, len(x) // 500)
            ax.quiver(
                x[::step], y[::step],
                ux[::step] * scale, uy[::step] * scale,
                angles="xy", scale_units="xy", scale=1,
                **kwargs,
            )

    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")

    return ax
