"""Energy functional assembly for moire relaxation.

Computes the total energy E = E_elastic + E_GSFE, its gradient, and Hessian.

The elastic Hessian is constant (precomputed once as sparse matrix).
The GSFE Hessian is vertex-diagonal (cheap per iteration).

For multi-layer systems, GSFE coupling exists at:
- The interface between the two stacks (always present)
- Between adjacent layers within each stack (intra-stack, when nlayer > 1)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .discretization import ConversionMatrices, Discretization, PinnedConstraints
from .gsfe import GSFESurface
from .lattice import MoireGeometry


@dataclass
class _GSFEPair:
    """Descriptor for a GSFE-coupled pair of layers."""

    gsfe: GSFESurface
    # DOF offsets for the two layers' ux and uy components
    layer_a_ox: int  # ux offset for layer A
    layer_b_ox: int  # ux offset for layer B
    layer_a_oy: int  # uy offset for layer A
    layer_b_oy: int  # uy offset for layer B
    # Natural stacking phase offset (in radians) — a *constant* offset
    # applied to the phase at every vertex (e.g. the ±2π/3 Bernal offset
    # for graphene homobilayer stacks, so u=0 corresponds to AB or BA).
    v_offset: float = 0.0
    w_offset: float = 0.0
    # If True, the position-dependent moire phase v0(r), w0(r) is added
    # to this pair's phase (only the twisted interface pair should have
    # this — the homobilayer pairs inside each flake are *untwisted*
    # and must NOT pick up the moire phase, else u=0 no longer sits at
    # Bernal and the solver ends up compensating with spurious
    # alternating ±u layer patterns).
    use_moire_phase: bool = True


def _flip_negative_eigenvalues_2x2(
    a: np.ndarray, b: np.ndarray, c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flip negative eigenvalues of per-vertex 2×2 symmetric matrices.

    Given the upper-triangular entries ``[[a, b], [b, c]]`` of N
    symmetric 2×2 matrices (one per vertex), return the entries of the
    matrices with all eigenvalues replaced by their absolute values.
    Eigenvectors are preserved; only the sign of the curvature in
    concave directions is flipped.

    This is an O(N) vectorized operation used by the modified-Newton
    solver to make the GSFE Hessian positive semi-definite at vertices
    that sit at stacking maxima (AA) or saddle points.
    """
    s = 0.5 * (a + c)           # mean eigenvalue
    diff = 0.5 * (a - c)
    d = np.sqrt(diff**2 + b**2)  # half-eigenvalue-spread (≥ 0)

    lam_max = s + d
    lam_min = s - d

    # Only modify vertices where at least one eigenvalue is negative.
    needs_fix = lam_min < 0
    if not np.any(needs_fix):
        return a, b, c

    # Modified eigenvalues
    lam_max_mod = np.abs(lam_max)
    lam_min_mod = np.abs(lam_min)

    s_mod = 0.5 * (lam_max_mod + lam_min_mod)
    d_mod = 0.5 * (lam_max_mod - lam_min_mod)

    # Reconstruct: a = s + d*cos(2θ), c = s - d*cos(2θ), b = d*sin(2θ)
    # cos(2θ) = diff/d, sin(2θ) = b/d
    ratio = np.where(d > 1e-30, d_mod / d, 0.0)

    a_out = a.copy()
    b_out = b.copy()
    c_out = c.copy()
    a_out[needs_fix] = s_mod[needs_fix] + ratio[needs_fix] * diff[needs_fix]
    c_out[needs_fix] = s_mod[needs_fix] - ratio[needs_fix] * diff[needs_fix]
    b_out[needs_fix] = ratio[needs_fix] * b[needs_fix]
    return a_out, b_out, c_out


class RelaxationEnergy:
    """Total energy functional for moire relaxation.

    Provides:
    - __call__(U) -> (E, grad)
    - hessian(U) -> sparse matrix
    - hessp(U, p) -> Hessian-vector product
    """

    def __init__(
        self,
        disc: Discretization,
        conv: ConversionMatrices,
        geometry: MoireGeometry,
        gsfe_interface: GSFESurface,
        K1: float,
        G1: float,
        K2: float,
        G2: float,
        nlayer1: int = 1,
        nlayer2: int = 1,
        gsfe_flake1: GSFESurface | None = None,
        gsfe_flake2: GSFESurface | None = None,
        I1_vect: np.ndarray | None = None,
        J1_vect: np.ndarray | None = None,
        I2_vect: np.ndarray | None = None,
        J2_vect: np.ndarray | None = None,
        constraints: PinnedConstraints | None = None,
    ):
        self.disc = disc
        self.conv = conv
        self.constraints = constraints
        self.geometry = geometry
        self.K1 = K1
        self.G1 = G1
        self.K2 = K2
        self.G2 = G2
        self.nlayer1 = nlayer1
        self.nlayer2 = nlayer2

        Nv = disc.mesh.n_vertices
        self.Nv = Nv
        nlayers_total = nlayer1 + nlayer2
        self._nlayers_total = nlayers_total

        # Precompute the unrelaxed stacking phases at mesh vertices
        x, y = disc.mesh.points[0], disc.mesh.points[1]
        self._v0, self._w0 = geometry.stacking_phases(x, y)

        # Stacking phase conversion matrix
        self._Mu1 = geometry.Mu1  # (2,2)

        # Precompute normalized area weights
        Suc = geometry.lattice.unit_cell_area
        self._area_v_norm = disc.vertex_areas / Suc
        self._area_t_norm = disc.triangle_areas / Suc

        # Cache diff matrices
        self._Dx = disc.diff_mat_x
        self._Dy = disc.diff_mat_y

        # --- Build list of all GSFE-coupled pairs ---
        I1 = I1_vect if I1_vect is not None else np.zeros(max(nlayer1 - 1, 0))
        J1 = J1_vect if J1_vect is not None else np.zeros(max(nlayer1 - 1, 0))
        I2 = I2_vect if I2_vect is not None else np.zeros(max(nlayer2 - 1, 0))
        J2 = J2_vect if J2_vect is not None else np.zeros(max(nlayer2 - 1, 0))

        self._gsfe_pairs: list[_GSFEPair] = []

        def _offsets(global_layer_idx):
            """Get (ox, oy) DOF offsets for a global layer index."""
            ox = global_layer_idx * Nv
            oy = nlayers_total * Nv + global_layer_idx * Nv
            return ox, oy

        # Interface between stacks: innermost layer of stack 1 <-> innermost of stack 2
        inner1_global = nlayer1 - 1
        inner2_global = nlayer1
        a_ox, a_oy = _offsets(inner1_global)
        b_ox, b_oy = _offsets(inner2_global)
        self._gsfe_pairs.append(_GSFEPair(
            gsfe=gsfe_interface,
            layer_a_ox=a_ox, layer_b_ox=b_ox,
            layer_a_oy=a_oy, layer_b_oy=b_oy,
        ))

        # Intra-stack 1: between layers k and k+1 (k = 0..nlayer1-2)
        if gsfe_flake1 is not None:
            for k in range(nlayer1 - 1):
                a_ox, a_oy = _offsets(k)
                b_ox, b_oy = _offsets(k + 1)
                self._gsfe_pairs.append(_GSFEPair(
                    gsfe=gsfe_flake1,
                    layer_a_ox=a_ox, layer_b_ox=b_ox,
                    layer_a_oy=a_oy, layer_b_oy=b_oy,
                    v_offset=2 * np.pi * I1[k],
                    w_offset=2 * np.pi * J1[k],
                    use_moire_phase=False,
                ))

        # Intra-stack 2: between layers k and k+1 (k = 0..nlayer2-2)
        if gsfe_flake2 is not None:
            for k in range(nlayer2 - 1):
                a_global = nlayer1 + k
                b_global = nlayer1 + k + 1
                a_ox, a_oy = _offsets(a_global)
                b_ox, b_oy = _offsets(b_global)
                self._gsfe_pairs.append(_GSFEPair(
                    gsfe=gsfe_flake2,
                    layer_a_ox=a_ox, layer_b_ox=b_ox,
                    layer_a_oy=a_oy, layer_b_oy=b_oy,
                    v_offset=2 * np.pi * I2[k],
                    w_offset=2 * np.pi * J2[k],
                    use_moire_phase=False,
                ))

        # Precompute the elastic Hessian (constant)
        self._H_elastic = self._build_elastic_hessian()

        self.eval_count = 0

    # --- Elastic Hessian (unchanged from before) ---

    def _build_elastic_hessian(self) -> sparse.csr_matrix:
        """Build the elastic Hessian (constant, precomputed once)."""
        Dx = self._Dx
        Dy = self._Dy
        W = sparse.diags(self._area_t_norm)
        n_sol = self.conv.n_sol
        Nv = self.Nv
        nlayers_total = self._nlayers_total

        WDx = W @ Dx
        WDy = W @ Dy
        DxTWDx = Dx.T @ WDx
        DxTWDy = Dx.T @ WDy
        DyTWDx = Dy.T @ WDx
        DyTWDy = Dy.T @ WDy

        all_rows, all_cols, all_data = [], [], []

        for stack_idx in range(2):
            nlayer = self.nlayer1 if stack_idx == 0 else self.nlayer2
            K = self.K1 if stack_idx == 0 else self.K2
            G = self.G1 if stack_idx == 0 else self.G2
            layer_offset = 0 if stack_idx == 0 else self.nlayer1

            H_xx = (K + G) * DxTWDx + G * DyTWDy
            H_xy = (K - G) * DxTWDy + G * DyTWDx
            H_yx = G * DxTWDy + (K - G) * DyTWDx
            H_yy = G * DxTWDx + (K + G) * DyTWDy

            for k in range(nlayer):
                layer_idx = layer_offset + k
                ox = layer_idx * Nv
                oy = nlayers_total * Nv + layer_idx * Nv

                for blk, r_off, c_off in [(H_xx, ox, ox), (H_xy, ox, oy),
                                           (H_yx, oy, ox), (H_yy, oy, oy)]:
                    coo = blk.tocoo()
                    all_rows.append(coo.row + r_off)
                    all_cols.append(coo.col + c_off)
                    all_data.append(coo.data)

        rows = np.concatenate(all_rows)
        cols = np.concatenate(all_cols)
        data = np.concatenate(all_data)
        return sparse.csr_matrix((data, (rows, cols)), shape=(n_sol, n_sol))

    # --- GSFE pair helpers ---

    def _pair_phases(self, pair: _GSFEPair, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute stacking phases for a GSFE pair."""
        Nv = self.Nv
        dux = U[pair.layer_a_ox:pair.layer_a_ox + Nv] - U[pair.layer_b_ox:pair.layer_b_ox + Nv]
        duy = U[pair.layer_a_oy:pair.layer_a_oy + Nv] - U[pair.layer_b_oy:pair.layer_b_oy + Nv]
        Mu = self._Mu1
        dv = -(Mu[0, 0] * dux + Mu[0, 1] * duy)
        dw = -(Mu[1, 0] * dux + Mu[1, 1] * duy)
        if pair.use_moire_phase:
            return self._v0 + dv + pair.v_offset, self._w0 + dw + pair.w_offset
        return dv + pair.v_offset, dw + pair.w_offset

    def _scatter_pair_gradient(self, pair: _GSFEPair, grad: np.ndarray,
                                fx: np.ndarray, fy: np.ndarray):
        """Add GSFE force contributions for a pair to gradient."""
        Nv = self.Nv
        grad[pair.layer_a_ox:pair.layer_a_ox + Nv] += fx
        grad[pair.layer_b_ox:pair.layer_b_ox + Nv] -= fx
        grad[pair.layer_a_oy:pair.layer_a_oy + Nv] += fy
        grad[pair.layer_b_oy:pair.layer_b_oy + Nv] -= fy

    # --- Energy, gradient, Hessian ---

    def _to_full(self, U: np.ndarray) -> np.ndarray:
        """Expand U from free-DOF space to full space if constrained."""
        if self.constraints is not None:
            return self.constraints.expand(U)
        return U

    def __call__(self, U: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute total energy and gradient.

        If constraints are set, U is in free-DOF space. The returned gradient
        is also in free-DOF space.
        """
        self.eval_count += 1
        U_full = self._to_full(U)

        # Elastic energy (quadratic: E = 0.5 * U^T H U, grad = H U)
        HU = self._H_elastic @ U_full
        E_total = 0.5 * U_full.dot(HU)
        grad = HU.copy()

        # GSFE energy for all pairs
        Mu = self._Mu1
        w_v = self._area_v_norm

        for pair in self._gsfe_pairs:
            v, w = self._pair_phases(pair, U_full)
            V = pair.gsfe(v, w)
            V_min = pair.gsfe.minimum_value
            E_total += float(np.sum((V - V_min) * w_v))

            dVdv = pair.gsfe.dv(v, w)
            dVdw = pair.gsfe.dw(v, w)
            fx = -(Mu[0, 0] * dVdv + Mu[1, 0] * dVdw) * w_v
            fy = -(Mu[0, 1] * dVdv + Mu[1, 1] * dVdw) * w_v
            self._scatter_pair_gradient(pair, grad, fx, fy)

        if self.constraints is not None:
            grad = self.constraints.project(grad)

        return E_total, grad

    def hessp(self, U: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Compute Hessian-vector product H(U) @ p.

        If constrained, U and p are in free-DOF space; result is also free-DOF space.
        """
        Nv = self.Nv
        Mu = self._Mu1
        w_v = self._area_v_norm

        U_full = self._to_full(U)

        if self.constraints is not None:
            p_full = self.constraints.expand_zeros(p)  # zeros at pinned for perturbation
        else:
            p_full = p

        result = self._H_elastic @ p_full

        for pair in self._gsfe_pairs:
            v, w = self._pair_phases(pair, U_full)
            d2v2 = pair.gsfe.d2v2(v, w)
            d2w2 = pair.gsfe.d2w2(v, w)
            d2vw = pair.gsfe.d2vw(v, w)

            # Extract perturbation for this pair (use full-space p)
            dpx = p_full[pair.layer_a_ox:pair.layer_a_ox + Nv] - p_full[pair.layer_b_ox:pair.layer_b_ox + Nv]
            dpy = p_full[pair.layer_a_oy:pair.layer_a_oy + Nv] - p_full[pair.layer_b_oy:pair.layer_b_oy + Nv]

            dp_v = -(Mu[0, 0] * dpx + Mu[0, 1] * dpy)
            dp_w = -(Mu[1, 0] * dpx + Mu[1, 1] * dpy)

            hv = d2v2 * dp_v + d2vw * dp_w
            hw = d2vw * dp_v + d2w2 * dp_w

            hfx = -(Mu[0, 0] * hv + Mu[1, 0] * hw) * w_v
            hfy = -(Mu[0, 1] * hv + Mu[1, 1] * hw) * w_v
            self._scatter_pair_gradient(pair, result, hfx, hfy)

        if self.constraints is not None:
            result = self.constraints.project(result)

        return result

    def hessian(self, U: np.ndarray, modified: bool = False) -> sparse.csr_matrix:
        """Compute the Hessian. If constrained, returns the free x free subblock.

        Parameters
        ----------
        U : ndarray
            Current DOF vector (free-space if constrained).
        modified : bool
            If True, flip negative eigenvalues of the per-vertex GSFE
            Hessian blocks so the returned matrix is positive definite
            (modulo the elastic part, which is always PSD). This enables
            Newton steps at non-minimum stacking sites (AA, saddle
            points) where the unmodified GSFE Hessian is indefinite.
        """
        U_full = self._to_full(U)
        H_full = self._hessian_full(U_full, modified=modified)
        if self.constraints is not None:
            return self.constraints.project_hessian(H_full)
        return H_full

    def _hessian_full(self, U_full: np.ndarray, modified: bool = False) -> sparse.csr_matrix:
        """Compute the full-space Hessian.

        When ``modified=True``, the per-vertex 2×2 GSFE Hessian blocks
        have their negative eigenvalues flipped to positive before
        assembly. This makes the total Hessian positive definite
        (elastic part is always PSD) without any global scalar damping,
        enabling well-conditioned Newton steps even at AA-stacking
        sites and saddle points where the GSFE curvature is concave.
        """
        Nv = self.Nv
        n_sol = self.conv.n_sol
        Mu = self._Mu1
        w_v = self._area_v_norm
        idx = np.arange(Nv)

        all_rows, all_cols, all_data = [], [], []

        for pair in self._gsfe_pairs:
            v, w = self._pair_phases(pair, U_full)
            d2v2 = pair.gsfe.d2v2(v, w)
            d2w2 = pair.gsfe.d2w2(v, w)
            d2vw = pair.gsfe.d2vw(v, w)

            H_dxdx = w_v * (Mu[0, 0]**2 * d2v2 + 2*Mu[0, 0]*Mu[1, 0]*d2vw + Mu[1, 0]**2 * d2w2)
            H_dxdy = w_v * (Mu[0, 0]*Mu[0, 1]*d2v2 + (Mu[0, 0]*Mu[1, 1]+Mu[0, 1]*Mu[1, 0])*d2vw + Mu[1, 0]*Mu[1, 1]*d2w2)
            H_dydy = w_v * (Mu[0, 1]**2 * d2v2 + 2*Mu[0, 1]*Mu[1, 1]*d2vw + Mu[1, 1]**2 * d2w2)

            if modified:
                H_dxdx, H_dxdy, H_dydy = _flip_negative_eigenvalues_2x2(
                    H_dxdx, H_dxdy, H_dydy,
                )

            # Place blocks for this pair (4 combinations of layer_a/layer_b x/y)
            x_offsets = [
                (pair.layer_a_ox, pair.layer_a_ox, +1, +1),
                (pair.layer_a_ox, pair.layer_b_ox, +1, -1),
                (pair.layer_b_ox, pair.layer_a_ox, -1, +1),
                (pair.layer_b_ox, pair.layer_b_ox, -1, -1),
            ]
            y_offsets = [
                (pair.layer_a_oy, pair.layer_a_oy, +1, +1),
                (pair.layer_a_oy, pair.layer_b_oy, +1, -1),
                (pair.layer_b_oy, pair.layer_a_oy, -1, +1),
                (pair.layer_b_oy, pair.layer_b_oy, -1, -1),
            ]
            xy_offsets = [
                (pair.layer_a_ox, pair.layer_a_oy, +1, +1),
                (pair.layer_a_ox, pair.layer_b_oy, +1, -1),
                (pair.layer_b_ox, pair.layer_a_oy, -1, +1),
                (pair.layer_b_ox, pair.layer_b_oy, -1, -1),
            ]

            for r_off, c_off, sr, sc in x_offsets:
                all_rows.append(idx + r_off)
                all_cols.append(idx + c_off)
                all_data.append(sr * sc * H_dxdx)

            for r_off, c_off, sr, sc in y_offsets:
                all_rows.append(idx + r_off)
                all_cols.append(idx + c_off)
                all_data.append(sr * sc * H_dydy)

            for r_off, c_off, sr, sc in xy_offsets:
                all_rows.append(idx + r_off)
                all_cols.append(idx + c_off)
                all_data.append(sr * sc * H_dxdy)
                # Symmetric yx = xy^T
                all_rows.append(idx + c_off)
                all_cols.append(idx + r_off)
                all_data.append(sr * sc * H_dxdy)

        rows = np.concatenate(all_rows)
        cols = np.concatenate(all_cols)
        data = np.concatenate(all_data)

        H_gsfe = sparse.csr_matrix((data, (rows, cols)), shape=(n_sol, n_sol))
        return self._H_elastic + H_gsfe

    def energy_maps(self, U: np.ndarray) -> dict:
        """Compute spatially resolved energy density maps."""
        Nv = self.Nv
        Dx, Dy = self._Dx, self._Dy
        Suc = self.geometry.lattice.unit_cell_area

        # GSFE maps for all pairs
        gsfe_maps = []
        for i, pair in enumerate(self._gsfe_pairs):
            v, w = self._pair_phases(pair, U)
            V = pair.gsfe(v, w)
            V_min = pair.gsfe.minimum_value
            gsfe_maps.append((V - V_min) / Suc)

        # Elastic energy per layer
        t2v = self.disc.triangle_to_vertex
        elastic_maps = {}
        for stack_idx, label in enumerate(["elastic_1", "elastic_2"]):
            nlayer = self.nlayer1 if stack_idx == 0 else self.nlayer2
            K = self.K1 if stack_idx == 0 else self.K2
            G = self.G1 if stack_idx == 0 else self.G2
            conv_x = self.conv.conv_x1 if stack_idx == 0 else self.conv.conv_x2
            conv_y = self.conv.conv_y1 if stack_idx == 0 else self.conv.conv_y2

            maps_k = np.zeros((nlayer, Nv))
            for k in range(nlayer):
                ux_k = conv_x[k * Nv : (k + 1) * Nv] @ U
                uy_k = conv_y[k * Nv : (k + 1) * Nv] @ U
                exx = Dx @ ux_k
                exy = Dy @ ux_k
                eyx = Dx @ uy_k
                eyy = Dy @ uy_k
                trace = exx + eyy
                e_el = 0.5 * K * trace**2 + 0.5 * G * ((exx - eyy)**2 + (exy + eyx)**2)
                maps_k[k] = (t2v @ e_el) / Suc
            elastic_maps[label] = maps_k

        return {
            "gsfe_interface": gsfe_maps[0],  # first pair is always the interface
            "gsfe_all": gsfe_maps,
            **elastic_maps,
        }
