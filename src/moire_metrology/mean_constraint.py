"""Linear equality constraints on the mean displacement of a subset of DOFs.

A ``MeanDisplacementConstraint`` enforces

    (1/|S|) · Σ_{i ∈ S} u_{c,i} = target_c

for a chosen layer or vertex subset S and each component c ∈ {x, y}.
Typical uses:

- Lifting the 2D in-plane translation null mode of a Neumann-boundary
  flake without pinning any particular vertex (a vertex pin localizes
  stacking at AA to that point, which biases downstream physics).
- Holding a layer's centre of mass at a prescribed target while other
  layers relax freely, or while a `lateral_offset` drives the relative
  phase between layers.
- Anchoring a rigid-body rotation mode when combined with other
  constraints in multi-layer stacks.

The constraint is enforced exactly via Lagrange multipliers inside the
Newton solver. Supported solver configurations: ``method="newton"``
with ``linear_solver="direct"``. For ``L-BFGS-B`` or
``pseudo_dynamics`` you currently need to use a soft spring-based
approximation instead (see ``SolverConfig`` docs).

Example
-------
::

    from moire_metrology import (
        RelaxationSolver, SolverConfig,
        MeanDisplacementConstraint,
    )

    cfg = SolverConfig(method="newton", linear_solver="direct")
    solver = RelaxationSolver(cfg)

    # Pin the mean displacement of the innermost top-flake layer.
    mdc = MeanDisplacementConstraint.from_layer(
        conv, layer_idx=0, target=(0.0, 0.0),
    )
    result = solver.solve(
        moire_interface=ifc, theta_twist=1.0,
        mesh=my_finite_mesh,
        constraints=pinned_substrate_constraints,
        mean_constraints=[mdc],
    )
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .discretization import ConversionMatrices, PinnedConstraints


@dataclass(frozen=True)
class MeanDisplacementConstraint:
    r"""Linear equality constraints on the mean of chosen DOFs.

    Enforces ``(1/|S|) Σ_{i ∈ S} u_{c,i} = target_c`` for each
    component ``c`` in ``components`` (typically ``("x", "y")``) and a
    vertex set ``S`` restricted to a single layer.

    Attributes
    ----------
    layer_idx : int
        Global layer index (0 = outermost of the top flake, etc.).
    vertex_indices : ndarray of int
        The vertex subset ``S``. Repeated vertices are not allowed.
    target : tuple[float, float]
        Target mean displacement ``(target_x, target_y)`` in nm.
    components : tuple[str, ...]
        Which components to constrain; default ``("x", "y")`` = 2
        scalar constraints. Pass a subset to constrain only one.

    Notes
    -----
    The coefficient on each DOF is ``1/|S|``, so ``target`` is a mean
    displacement (not a sum). Passing ``target=(0.0, 0.0)`` with both
    components is the generic null-mode remover for a Neumann flake.
    """

    layer_idx: int
    vertex_indices: np.ndarray
    target: tuple[float, float] = (0.0, 0.0)
    components: tuple[str, ...] = ("x", "y")

    def __post_init__(self):
        vi = np.asarray(self.vertex_indices, dtype=np.int64)
        if vi.ndim != 1:
            raise ValueError("vertex_indices must be 1-D")
        if np.any(vi < 0):
            raise ValueError("vertex_indices must be non-negative")
        if len(np.unique(vi)) != len(vi):
            raise ValueError("vertex_indices must be unique")
        object.__setattr__(self, "vertex_indices", vi)
        for c in self.components:
            if c not in ("x", "y"):
                raise ValueError(
                    f"components entries must be 'x' or 'y', got {c!r}"
                )

    @classmethod
    def from_layer(
        cls,
        conv: ConversionMatrices,
        layer_idx: int,
        target: tuple[float, float] = (0.0, 0.0),
        components: tuple[str, ...] = ("x", "y"),
    ) -> MeanDisplacementConstraint:
        """Constrain the mean displacement over *all* vertices of a layer."""
        if not (0 <= layer_idx < conv.nlayer1 + conv.nlayer2):
            raise ValueError(
                f"layer_idx={layer_idx} out of range "
                f"[0, {conv.nlayer1 + conv.nlayer2})"
            )
        return cls(
            layer_idx=layer_idx,
            vertex_indices=np.arange(conv.n_vertices, dtype=np.int64),
            target=target,
            components=components,
        )

    # --- Matrix construction ---

    @property
    def n_rows(self) -> int:
        """Number of scalar constraint rows (one per component)."""
        return len(self.components)

    def build_matrix(
        self,
        conv: ConversionMatrices,
    ) -> tuple[sparse.csr_matrix, np.ndarray]:
        """Assemble the sparse constraint matrix and target vector.

        Returns
        -------
        B : csr_matrix, shape (n_rows, n_sol)
            Constraint matrix in full-DOF space, with ``B @ U = target``
            enforcing the mean-displacement equality.
        t : ndarray, shape (n_rows,)
            Target vector.
        """
        Nv = conv.n_vertices
        nlayers_total = conv.nlayer1 + conv.nlayer2
        n_sol = conv.n_sol
        S = len(self.vertex_indices)
        if S == 0:
            raise ValueError("vertex_indices is empty")
        coeff = 1.0 / float(S)

        rows = []
        cols = []
        data = []
        targets = []
        # Layer DOF offsets:
        ox = self.layer_idx * Nv
        oy = nlayers_total * Nv + self.layer_idx * Nv
        for r, comp in enumerate(self.components):
            offset = ox if comp == "x" else oy
            cols_r = offset + self.vertex_indices
            rows.append(np.full(S, r, dtype=np.int64))
            cols.append(cols_r)
            data.append(np.full(S, coeff))
            targets.append(self.target[0] if comp == "x" else self.target[1])

        rows_a = np.concatenate(rows)
        cols_a = np.concatenate(cols)
        data_a = np.concatenate(data)
        B = sparse.csr_matrix(
            (data_a, (rows_a, cols_a)),
            shape=(self.n_rows, n_sol),
        )
        t = np.asarray(targets, dtype=float)
        return B, t


@dataclass(frozen=True)
class RotationConstraint:
    r"""Linear equality constraint preventing net in-plane rotation.

    Enforces

        Σ_{i ∈ S} (x̃_i · u_{y,i} − ỹ_i · u_{x,i}) = 0

    where x̃, ỹ are vertex positions relative to the centroid of S.
    This kills the rigid-rotation null mode without constraining
    local deformations, complementing ``MeanDisplacementConstraint``
    which kills the translation null modes.

    Pass in ``mean_constraints=[mdc, rot]`` alongside the MDC.
    """

    layer_idx: int
    vertex_indices: np.ndarray
    mesh_points: np.ndarray  # (2, Nv_total) — full mesh points array

    @classmethod
    def from_layer(
        cls,
        conv: ConversionMatrices,
        mesh_points: np.ndarray,
        layer_idx: int = 0,
    ) -> "RotationConstraint":
        """Constrain net rotation over all vertices of a layer."""
        return cls(
            layer_idx=layer_idx,
            vertex_indices=np.arange(conv.n_vertices, dtype=np.int64),
            mesh_points=mesh_points,
        )

    @property
    def n_rows(self) -> int:
        return 1

    def build_matrix(
        self,
        conv: ConversionMatrices,
    ) -> tuple[sparse.csr_matrix, np.ndarray]:
        """Build the 1×n_sol constraint row and target (=0)."""
        Nv = conv.n_vertices
        nlayers_total = conv.nlayer1 + conv.nlayer2
        n_sol = conv.n_sol
        vi = self.vertex_indices
        S = len(vi)

        # Centroid-relative coordinates
        x = self.mesh_points[0, vi]
        y = self.mesh_points[1, vi]
        xc = x - x.mean()
        yc = y - y.mean()

        # DOF offsets for this layer
        ox = self.layer_idx * Nv        # ux start
        oy = nlayers_total * Nv + self.layer_idx * Nv  # uy start

        # Row 0: Σ (x̃_i · u_yi − ỹ_i · u_xi) = 0
        # Normalize by Σ(x̃² + ỹ²) so the constraint has O(1) scale
        norm = np.sum(xc**2 + yc**2)
        if norm < 1e-30:
            raise ValueError("Vertices are co-located; cannot define rotation")

        rows = np.zeros(2 * S, dtype=np.int64)  # all row 0
        cols = np.concatenate([ox + vi, oy + vi])
        data = np.concatenate([-yc / norm, xc / norm])

        B = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(1, n_sol),
        )
        t = np.array([0.0])
        return B, t


@dataclass(frozen=True)
class PeriodicPairConstraint:
    r"""Linear equality constraints enforcing ``u_i = u_j`` for vertex pairs.

    For each ``(i, j)`` in ``pairs`` and each component ``c ∈ components``
    (typically ``("x", "y")``) the constraint

        u_{c,i} - u_{c,j} = 0

    is added as a single row. Used to impose kinematic periodicity across
    a cut in an otherwise open mesh — e.g. matching the two long edges
    of a finite parallelogram mesh into a cylinder, or matching the
    three pairs of opposite edges of a Wigner-Seitz hexagonal supercell
    (see :func:`moire_metrology.mesh.generate_hex_periodic_mesh` and
    :func:`moire_metrology.mesh.identify_hex_periodic_boundary`).

    Stress continuity is not enforced by the constraint alone; any
    residual traction mismatch across the cut is localized to within
    one element of the matched edges, which is acceptable when the cut
    is far from the physics of interest or when the lattice is itself
    periodic (so the cut sits at a true symmetry plane of the problem).

    Attributes
    ----------
    layer_idx : int
        Global layer index (0 = outermost of the top flake).
    pairs : ndarray of int, shape (N, 2)
        Vertex index pairs to identify. Each pair contributes
        ``len(components)`` scalar constraints.
    components : tuple[str, ...]
        Which components to match; default ``("x", "y")``.
    """

    layer_idx: int
    pairs: np.ndarray
    components: tuple[str, ...] = ("x", "y")

    def __post_init__(self):
        pr = np.asarray(self.pairs, dtype=np.int64)
        if pr.ndim != 2 or pr.shape[1] != 2:
            raise ValueError("pairs must have shape (N, 2)")
        if np.any(pr < 0):
            raise ValueError("vertex indices must be non-negative")
        if np.any(pr[:, 0] == pr[:, 1]):
            raise ValueError("each pair must reference two distinct vertices")
        object.__setattr__(self, "pairs", pr)
        for c in self.components:
            if c not in ("x", "y"):
                raise ValueError(
                    f"components entries must be 'x' or 'y', got {c!r}"
                )

    @property
    def n_rows(self) -> int:
        return len(self.pairs) * len(self.components)

    def build_matrix(
        self,
        conv: ConversionMatrices,
    ) -> tuple[sparse.csr_matrix, np.ndarray]:
        Nv = conv.n_vertices
        nlayers_total = conv.nlayer1 + conv.nlayer2
        n_sol = conv.n_sol
        N = len(self.pairs)
        C = len(self.components)

        ox = self.layer_idx * Nv
        oy = nlayers_total * Nv + self.layer_idx * Nv

        rows = np.empty(2 * N * C, dtype=np.int64)
        cols = np.empty(2 * N * C, dtype=np.int64)
        data = np.empty(2 * N * C, dtype=float)

        k = 0
        for c_idx, comp in enumerate(self.components):
            base = ox if comp == "x" else oy
            for pair_idx, (i, j) in enumerate(self.pairs):
                row = c_idx * N + pair_idx
                rows[k] = row
                cols[k] = base + i
                data[k] = 1.0
                k += 1
                rows[k] = row
                cols[k] = base + j
                data[k] = -1.0
                k += 1

        B = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(N * C, n_sol),
        )
        t = np.zeros(N * C)
        return B, t


def stack_mean_constraints(
    mean_constraints: list[
        MeanDisplacementConstraint
        | RotationConstraint
        | PeriodicPairConstraint
    ],
    conv: ConversionMatrices,
    pinned_constraints: PinnedConstraints | None = None,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Combine multiple MDCs into a single (B, t) pair in free-DOF space.

    If ``pinned_constraints`` is provided, the returned ``B`` is
    restricted to free-DOF columns and the routine checks that no
    constrained DOF coincides with a pinned DOF (which would make the
    mean constraint redundant or inconsistent with the pin).

    Returns
    -------
    B_free : csr_matrix, shape (m_total, n_free)
    t : ndarray, shape (m_total,)
    """
    if not mean_constraints:
        raise ValueError("mean_constraints is empty")
    B_blocks = []
    t_blocks = []
    for mc in mean_constraints:
        B, t = mc.build_matrix(conv)
        B_blocks.append(B)
        t_blocks.append(t)
    B_full = sparse.vstack(B_blocks, format="csr")
    t_full = np.concatenate(t_blocks)

    if pinned_constraints is not None:
        # Check: no constraint DOF is a pinned DOF.
        pinned_set = set(pinned_constraints.pinned_indices.tolist())
        # Nonzero columns of B_full:
        active_cols = np.unique(B_full.indices)
        bad = [c for c in active_cols if c in pinned_set]
        if bad:
            raise ValueError(
                f"Mean constraint touches pinned DOF indices {bad[:5]}"
                f"{'...' if len(bad) > 5 else ''}. Pinned DOFs are "
                "already fixed; constraining their mean is inconsistent."
            )
        # Restrict to free-DOF columns.
        B_free = B_full[:, pinned_constraints.free_indices]
        return B_free.tocsr(), t_full
    return B_full, t_full
