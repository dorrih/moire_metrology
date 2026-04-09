"""Support-domain helpers for spatial strain extraction.

A polynomial registry fit is only trustworthy inside the convex hull of
its input points. Outside the hull a high-degree polynomial extrapolates
with rapid growth, which would otherwise produce nonsense displacements
or strain values at mesh vertices that lie past the data extent. The
helpers here let the caller mask query points to the data support.
"""

from __future__ import annotations

import numpy as np


def convex_hull_mask(
    data_x: np.ndarray,
    data_y: np.ndarray,
    query_x: np.ndarray,
    query_y: np.ndarray,
) -> np.ndarray:
    """Boolean mask of query points inside the convex hull of input data.

    Parameters
    ----------
    data_x, data_y : ndarray
        Coordinates of the points used to fit a registry polynomial.
    query_x, query_y : ndarray
        Query coordinates (any common shape).

    Returns
    -------
    inside : ndarray of bool
        Same shape as ``query_x``. True where the point lies inside (or
        on the boundary of) the convex hull of the input data.
    """
    from matplotlib.path import Path as MplPath
    from scipy.spatial import ConvexHull

    data_x = np.asarray(data_x).ravel()
    data_y = np.asarray(data_y).ravel()
    points = np.column_stack([data_x, data_y])

    hull = ConvexHull(points)
    hull_path = MplPath(points[hull.vertices])

    qx = np.asarray(query_x)
    qy = np.asarray(query_y)
    shape = qx.shape
    flat = np.column_stack([qx.ravel(), qy.ravel()])
    inside = hull_path.contains_points(flat)
    return inside.reshape(shape)
