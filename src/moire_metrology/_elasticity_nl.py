"""Geometrically nonlinear (St.-Venant–Kirchhoff) elastic term.

Replaces the linearized Cauchy strain ``ε = ½(∇u + ∇u^T)`` with the
Green–Lagrange strain ``E = ½(F^T F − I)``, where ``F = I + ∇u``.
Rigid rotations give ``F = R`` and therefore ``E ≡ 0`` — the
rotation-invariance that Cauchy strain fails at finite rotation.

Convention.  The existing code parameterizes the linear elastic
energy by ``(K, G)`` such that the Cauchy density is

    ψ_C = ½(λ + 2μ) (u_x,x² + u_y,y²) + λ u_x,x u_y,y
          + ½ μ (u_x,y² + u_y,x²) + μ u_x,y u_y,x,

with ``λ = K − G`` and ``μ = G``. The St.-Venant–Kirchhoff density
uses the same ``(λ, μ)``:

    ψ_SVK = ½ λ tr(E)² + μ (E : E),

so that ``ψ_SVK → ψ_C`` as ``|∇u| → 0``.

Implementation layout.  All quantities are evaluated per triangle in a
vectorized way.  The per-triangle elastic tangent is 6 × 6 (three
nodes × two components) and is assembled by stacking (row, col, data)
triples from every layer and every triangle.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .discretization import ConversionMatrices, Discretization
from .lattice import MoireGeometry


class _GLElastic:
    """Per-layer St.-Venant–Kirchhoff assembler.

    Holds the immutable mesh / shape-gradient data and provides three
    operations, each applied layer-by-layer inside a single call:

    - ``energy_grad(U_full, grad_out) -> float``
      Adds the elastic gradient into ``grad_out`` in place, returns
      the elastic energy.
    - ``hessp(U_full, p_full, result_out)``
      Adds the elastic Hessian-vector product into ``result_out`` in
      place.
    - ``hessian_triples(U_full) -> (rows, cols, data)``
      Returns sparse-COO triples for the elastic Hessian, to be
      merged with the GSFE triples before the full matrix is built.
    """

    def __init__(
        self,
        disc: Discretization,
        conv: ConversionMatrices,
        geometry: MoireGeometry,
        K1: float,
        G1: float,
        K2: float,
        G2: float,
        nlayer1: int,
        nlayer2: int,
    ):
        self.disc = disc
        self.K1, self.G1 = K1, G1
        self.K2, self.G2 = K2, G2
        self.nlayer1 = nlayer1
        self.nlayer2 = nlayer2
        self.Nv = disc.mesh.n_vertices
        self.Nt = disc.mesh.n_triangles
        self._nlayers_total = nlayer1 + nlayer2
        self.n_sol = conv.n_sol

        # Per-triangle shape gradients, shape (Nt, 3) each. Extract from
        # the sparse diff matrices — each triangle row has three nonzeros
        # (one per node), in the order defined by mesh.triangles.
        self.tris = disc.mesh.triangles  # (Nt, 3)
        Dx = disc.diff_mat_x.tocsr()
        Dy = disc.diff_mat_y.tocsr()
        bx = np.zeros((self.Nt, 3))
        by = np.zeros((self.Nt, 3))
        for k in range(self.Nt):
            for j in range(3):
                bx[k, j] = Dx[k, self.tris[k, j]]
                by[k, j] = Dy[k, self.tris[k, j]]
        self.bx = bx  # (Nt, 3)
        self.by = by
        # Use unit-cell-normalized triangle areas to match the Cauchy path's
        # energy convention (meV/uc × number-of-unit-cells = meV).
        Suc = geometry.lattice.unit_cell_area
        self.A_tri = disc.triangle_areas / Suc  # (Nt,)

        # Precomputed DOF offsets per layer.
        self._ox = [li * self.Nv for li in range(self._nlayers_total)]
        self._oy = [
            self._nlayers_total * self.Nv + li * self.Nv
            for li in range(self._nlayers_total)
        ]

    # --- Per-layer (K, G) resolver ---

    def _layer_params(self, layer_idx: int) -> tuple[float, float]:
        """Return (λ, μ) = (K − G, G) for the stack containing this layer."""
        if layer_idx < self.nlayer1:
            return self.K1 - self.G1, self.G1
        return self.K2 - self.G2, self.G2

    # --- Per-triangle kinematic quantities ---

    def _triangle_F(
        self, u_x_nodes: np.ndarray, u_y_nodes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (F_11, F_12, F_21, F_22) per triangle for one layer.

        F = I + ∇u; linear-triangle ∇u is constant per triangle.
        """
        bx = self.bx
        by = self.by
        t = self.tris
        ux_t = u_x_nodes[t]  # (Nt, 3)
        uy_t = u_y_nodes[t]
        uxx = np.einsum("kj,kj->k", bx, ux_t)
        uxy = np.einsum("kj,kj->k", by, ux_t)
        uyx = np.einsum("kj,kj->k", bx, uy_t)
        uyy = np.einsum("kj,kj->k", by, uy_t)
        return 1.0 + uxx, uxy, uyx, 1.0 + uyy

    # --- Core per-layer operations ---

    def _layer_energy_grad(
        self, U_full: np.ndarray, layer_idx: int, grad_out: np.ndarray,
    ) -> float:
        """Accumulate elastic energy + gradient for one layer. Returns energy."""
        Nv = self.Nv
        ox = self._ox[layer_idx]
        oy = self._oy[layer_idx]
        lam, mu = self._layer_params(layer_idx)

        u_x = U_full[ox:ox + Nv]
        u_y = U_full[oy:oy + Nv]

        F11, F12, F21, F22 = self._triangle_F(u_x, u_y)
        E11 = 0.5 * (F11 * F11 + F21 * F21 - 1.0)
        E22 = 0.5 * (F12 * F12 + F22 * F22 - 1.0)
        E12 = 0.5 * (F11 * F12 + F21 * F22)
        trE = E11 + E22
        S11 = lam * trE + 2.0 * mu * E11
        S22 = lam * trE + 2.0 * mu * E22
        S12 = 2.0 * mu * E12

        psi = 0.5 * lam * trE * trE + mu * (E11 * E11 + E22 * E22
                                            + 2.0 * E12 * E12)
        E_layer = float(np.sum(psi * self.A_tri))

        # First Piola-Kirchhoff P = F · S, as per-triangle arrays.
        P11 = F11 * S11 + F12 * S12
        P12 = F11 * S12 + F12 * S22
        P21 = F21 * S11 + F22 * S12
        P22 = F21 * S12 + F22 * S22

        # Scatter to nodal gradient using linear-triangle shape gradients.
        A = self.A_tri
        t = self.tris
        bx = self.bx
        by = self.by
        # Gradient contribution at triangle k, node j:
        #   grad_u_x^j += A_k * (P11_k bx_{k,j} + P12_k by_{k,j})
        #   grad_u_y^j += A_k * (P21_k bx_{k,j} + P22_k by_{k,j})
        gx_node = A[:, None] * (P11[:, None] * bx + P12[:, None] * by)
        gy_node = A[:, None] * (P21[:, None] * bx + P22[:, None] * by)
        np.add.at(grad_out, ox + t, gx_node)
        np.add.at(grad_out, oy + t, gy_node)

        return E_layer

    def _layer_hessp(
        self, U_full: np.ndarray, p_full: np.ndarray,
        layer_idx: int, result_out: np.ndarray,
    ) -> None:
        """Add elastic Hessian-vector product for one layer into result_out."""
        Nv = self.Nv
        ox = self._ox[layer_idx]
        oy = self._oy[layer_idx]
        lam, mu = self._layer_params(layer_idx)

        u_x = U_full[ox:ox + Nv]
        u_y = U_full[oy:oy + Nv]
        px = p_full[ox:ox + Nv]
        py = p_full[oy:oy + Nv]

        F11, F12, F21, F22 = self._triangle_F(u_x, u_y)
        E11 = 0.5 * (F11 * F11 + F21 * F21 - 1.0)
        E22 = 0.5 * (F12 * F12 + F22 * F22 - 1.0)
        E12 = 0.5 * (F11 * F12 + F21 * F22)
        trE = E11 + E22
        S11 = lam * trE + 2.0 * mu * E11
        S22 = lam * trE + 2.0 * mu * E22
        S12 = 2.0 * mu * E12

        # dF from p (per triangle)
        bx = self.bx
        by = self.by
        t = self.tris
        px_t = px[t]
        py_t = py[t]
        dF11 = np.einsum("kj,kj->k", bx, px_t)
        dF12 = np.einsum("kj,kj->k", by, px_t)
        dF21 = np.einsum("kj,kj->k", bx, py_t)
        dF22 = np.einsum("kj,kj->k", by, py_t)

        # dE = sym(F^T dF)
        dE11 = F11 * dF11 + F21 * dF21
        dE22 = F12 * dF12 + F22 * dF22
        dE12 = 0.5 * (F11 * dF12 + F21 * dF22 + F12 * dF11 + F22 * dF21)
        dtrE = dE11 + dE22
        dS11 = lam * dtrE + 2.0 * mu * dE11
        dS22 = lam * dtrE + 2.0 * mu * dE22
        dS12 = 2.0 * mu * dE12

        # dP = dF · S + F · dS
        dP11 = dF11 * S11 + dF12 * S12 + F11 * dS11 + F12 * dS12
        dP12 = dF11 * S12 + dF12 * S22 + F11 * dS12 + F12 * dS22
        dP21 = dF21 * S11 + dF22 * S12 + F21 * dS11 + F22 * dS12
        dP22 = dF21 * S12 + dF22 * S22 + F21 * dS12 + F22 * dS22

        A = self.A_tri
        gx_node = A[:, None] * (dP11[:, None] * bx + dP12[:, None] * by)
        gy_node = A[:, None] * (dP21[:, None] * bx + dP22[:, None] * by)
        np.add.at(result_out, ox + t, gx_node)
        np.add.at(result_out, oy + t, gy_node)

    def _layer_hessian_triples(
        self, U_full: np.ndarray, layer_idx: int,
        rows_list: list, cols_list: list, data_list: list,
    ) -> None:
        """Append per-layer sparse Hessian (row, col, data) triples."""
        Nv = self.Nv
        ox = self._ox[layer_idx]
        oy = self._oy[layer_idx]
        lam, mu = self._layer_params(layer_idx)

        u_x = U_full[ox:ox + Nv]
        u_y = U_full[oy:oy + Nv]
        F11, F12, F21, F22 = self._triangle_F(u_x, u_y)
        E11 = 0.5 * (F11 * F11 + F21 * F21 - 1.0)
        E22 = 0.5 * (F12 * F12 + F22 * F22 - 1.0)
        E12 = 0.5 * (F11 * F12 + F21 * F22)
        trE = E11 + E22
        S11 = lam * trE + 2.0 * mu * E11
        S22 = lam * trE + 2.0 * mu * E22
        S12 = 2.0 * mu * E12

        # Voigt B matrix, shape (Nt, 3, 6). Column order: [ux0, ux1, ux2,
        # uy0, uy1, uy2]. Rows: [E_11, E_22, γ_12 ≡ 2 E_12].
        bx = self.bx
        by = self.by
        Nt = self.Nt
        B = np.zeros((Nt, 3, 6))
        B[:, 0, 0:3] = F11[:, None] * bx
        B[:, 0, 3:6] = F21[:, None] * bx
        B[:, 1, 0:3] = F12[:, None] * by
        B[:, 1, 3:6] = F22[:, None] * by
        B[:, 2, 0:3] = F11[:, None] * by + F12[:, None] * bx
        B[:, 2, 3:6] = F21[:, None] * by + F22[:, None] * bx

        # Voigt material matrix C (SVK, plane-strain-like): 3x3, acting on
        # [E_11, E_22, γ_12]. For energy ψ = ½ ê^T C ê where ê uses
        # engineering shear, the shear block is μ (not 2μ).
        # ψ = ½ λ (E_11 + E_22)^2 + μ (E_11² + E_22² + 2 E_12²)
        #    = ½ λ (E_11 + E_22)^2 + μ (E_11² + E_22²) + ½ μ γ_12²
        # So C = [[λ+2μ, λ, 0], [λ, λ+2μ, 0], [0, 0, μ]].
        C = np.array([
            [lam + 2.0 * mu, lam, 0.0],
            [lam, lam + 2.0 * mu, 0.0],
            [0.0, 0.0, mu],
        ])

        # Material tangent: K_mat = A · B^T · C · B   (Nt, 6, 6)
        # Use einsum for speed.
        CB = np.einsum("ij,kjm->kim", C, B)              # (Nt, 3, 6)
        K_mat = np.einsum("kji,kjm->kim", B, CB)         # (Nt, 6, 6)
        K_mat *= self.A_tri[:, None, None]

        # Geometric (initial-stress) stiffness:
        # K_geom(i,j) = A · (bx_i S_11 bx_j + bx_i S_12 by_j
        #                   + by_i S_12 bx_j + by_i S_22 by_j) · δ_{component}
        # Block-diagonal in (ux-ux) and (uy-uy); zero in (ux-uy).
        GS = (
            S11[:, None, None] * bx[:, :, None] * bx[:, None, :]
            + S12[:, None, None] * (bx[:, :, None] * by[:, None, :]
                                    + by[:, :, None] * bx[:, None, :])
            + S22[:, None, None] * by[:, :, None] * by[:, None, :]
        )  # (Nt, 3, 3)
        GS *= self.A_tri[:, None, None]

        K_tri = K_mat.copy()
        K_tri[:, 0:3, 0:3] += GS
        K_tri[:, 3:6, 3:6] += GS

        # Scatter to full-index (rows, cols, data).
        # DOF indices for this layer's triangle:
        # local 0..2 -> ox + tris[k], local 3..5 -> oy + tris[k]
        t = self.tris
        global_dofs = np.empty((Nt, 6), dtype=np.int64)
        global_dofs[:, 0:3] = ox + t
        global_dofs[:, 3:6] = oy + t

        rows = np.broadcast_to(global_dofs[:, :, None], (Nt, 6, 6)).reshape(-1)
        cols = np.broadcast_to(global_dofs[:, None, :], (Nt, 6, 6)).reshape(-1)
        data = K_tri.reshape(-1)
        rows_list.append(rows)
        cols_list.append(cols)
        data_list.append(data)

    # --- Public API called by RelaxationEnergy ---

    def energy_grad(self, U_full: np.ndarray, grad_out: np.ndarray) -> float:
        """Accumulate elastic energy + gradient across all layers."""
        E = 0.0
        for li in range(self._nlayers_total):
            E += self._layer_energy_grad(U_full, li, grad_out)
        return E

    def hessp(
        self, U_full: np.ndarray, p_full: np.ndarray, result_out: np.ndarray,
    ) -> None:
        for li in range(self._nlayers_total):
            self._layer_hessp(U_full, p_full, li, result_out)

    def hessian_sparse(self, U_full: np.ndarray) -> sparse.csr_matrix:
        rows_list: list = []
        cols_list: list = []
        data_list: list = []
        for li in range(self._nlayers_total):
            self._layer_hessian_triples(
                U_full, li, rows_list, cols_list, data_list,
            )
        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        data = np.concatenate(data_list)
        return sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(self.n_sol, self.n_sol),
        )
