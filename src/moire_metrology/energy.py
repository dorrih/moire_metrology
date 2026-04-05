"""Energy functional assembly for moire relaxation.

Computes the total energy E = E_elastic + E_GSFE, its gradient, and Hessian
with respect to the displacement field.

The elastic energy density at each triangle is:
    e_el = (1/Suc) * [0.5*K*(exx+eyy)^2 + 0.5*G*((exx-eyy)^2 + (exy+eyx)^2)]

The GSFE energy density at each vertex is:
    e_gsfe = (1/Suc) * (V(v,w) - V_min)

The elastic Hessian is constant (energy is quadratic in U) and precomputed once.
The GSFE Hessian is vertex-diagonal and recomputed each iteration.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .discretization import ConversionMatrices, PeriodicDiscretization
from .gsfe import GSFESurface
from .lattice import MoireGeometry


class RelaxationEnergy:
    """Total energy functional for moire relaxation.

    Provides:
    - __call__(U) -> (E, grad)  for scipy.optimize.minimize with jac=True
    - hessian(U) -> sparse matrix  for trust-region methods
    - hessp(U, p) -> ndarray  for Hessian-vector product methods

    Parameters
    ----------
    disc : PeriodicDiscretization
    conv : ConversionMatrices
    geometry : MoireGeometry
    gsfe_interface : GSFESurface
    K1, G1, K2, G2 : float
        Elastic constants (meV/uc).
    nlayer1, nlayer2 : int
    gsfe_flake1, gsfe_flake2 : GSFESurface or None
    I1_vect, J1_vect, I2_vect, J2_vect : ndarray or None
    """

    def __init__(
        self,
        disc: PeriodicDiscretization,
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
    ):
        self.disc = disc
        self.conv = conv
        self.geometry = geometry
        self.gsfe_interface = gsfe_interface
        self.K1 = K1
        self.G1 = G1
        self.K2 = K2
        self.G2 = G2
        self.nlayer1 = nlayer1
        self.nlayer2 = nlayer2
        self.gsfe_flake1 = gsfe_flake1
        self.gsfe_flake2 = gsfe_flake2

        Nv = disc.mesh.n_vertices
        self.Nv = Nv
        self.I1 = I1_vect if I1_vect is not None else np.zeros(max(nlayer1 - 1, 0))
        self.J1 = J1_vect if J1_vect is not None else np.zeros(max(nlayer1 - 1, 0))
        self.I2 = I2_vect if I2_vect is not None else np.zeros(max(nlayer2 - 1, 0))
        self.J2 = J2_vect if J2_vect is not None else np.zeros(max(nlayer2 - 1, 0))

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

        # Precompute interface layer offsets for direct indexing
        # For the interface: layer1 innermost = (nlayer1-1), layer2 innermost = 0
        nlayers_total = nlayer1 + nlayer2
        inner1_idx = (nlayer1 - 1)  # index within stack 1
        inner2_idx = nlayer1  # index within full layer list (stack2 layer 0)

        # Offsets into U for the interface layers' x and y components
        self._intf1_ox = inner1_idx * Nv       # ux of inner layer 1
        self._intf2_ox = inner2_idx * Nv       # ux of inner layer 2
        self._intf1_oy = nlayers_total * Nv + inner1_idx * Nv  # uy of inner layer 1
        self._intf2_oy = nlayers_total * Nv + inner2_idx * Nv  # uy of inner layer 2
        self._nlayers_total = nlayers_total

        # Precompute the elastic Hessian (constant)
        self._H_elastic = self._build_elastic_hessian()

        self.eval_count = 0

    def _build_elastic_hessian(self) -> sparse.csr_matrix:
        """Build the elastic Hessian (constant, precomputed once).

        The elastic energy for one layer is quadratic in U:
            E_el = 0.5 * U^T H_el U
        so the Hessian is exactly H_el.

        For strain components exx=Dx@ux, exy=Dy@ux, eyx=Dx@uy, eyy=Dy@uy:
            E = sum_t w_t * [0.5*K*(exx+eyy)^2 + 0.5*G*((exx-eyy)^2 + (exy+eyx)^2)]

        The Hessian blocks (per layer) are:
            H_ux_ux = (K+G)*Dx^T W Dx + G*Dy^T W Dy
            H_ux_uy = (K-G)*Dx^T W Dy + G*Dy^T W Dx
            H_uy_ux = G*Dx^T W Dy + (K-G)*Dy^T W Dx
            H_uy_uy = G*Dx^T W Dx + (K+G)*Dy^T W Dy

        Since the conversion matrices are simple identity blocks at offsets,
        we place the Hessian blocks directly into the correct positions.
        """
        Dx = self._Dx
        Dy = self._Dy
        W = sparse.diags(self._area_t_norm)
        n_sol = self.conv.n_sol
        Nv = self.Nv
        nlayers_total = self.nlayer1 + self.nlayer2

        # Precompute the 4 Hessian sub-blocks in (Nv, Nv) vertex space
        WDx = W @ Dx
        WDy = W @ Dy
        DxTWDx = Dx.T @ WDx
        DxTWDy = Dx.T @ WDy
        DyTWDx = Dy.T @ WDx
        DyTWDy = Dy.T @ WDy

        # Build the full Hessian by placing blocks at the correct offsets
        # Solution vector layout: [ux_layer0..ux_layerN, uy_layer0..uy_layerN]
        # Each layer block occupies Nv entries
        blocks = []
        block_rows = []
        block_cols = []

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
                # ux block offset = layer_idx * Nv
                # uy block offset = nlayers_total * Nv + layer_idx * Nv
                ox = layer_idx * Nv
                oy = nlayers_total * Nv + layer_idx * Nv

                # H[ox:ox+Nv, ox:ox+Nv] += H_xx (ux-ux)
                blocks.append(H_xx)
                block_rows.append(ox)
                block_cols.append(ox)

                # H[ox:ox+Nv, oy:oy+Nv] += H_xy (ux-uy)
                blocks.append(H_xy)
                block_rows.append(ox)
                block_cols.append(oy)

                # H[oy:oy+Nv, ox:ox+Nv] += H_yx (uy-ux)
                blocks.append(H_yx)
                block_rows.append(oy)
                block_cols.append(ox)

                # H[oy:oy+Nv, oy:oy+Nv] += H_yy (uy-uy)
                blocks.append(H_yy)
                block_rows.append(oy)
                block_cols.append(oy)

        # Assemble into full sparse matrix using COO format
        all_rows = []
        all_cols = []
        all_data = []

        for blk, r_off, c_off in zip(blocks, block_rows, block_cols):
            coo = blk.tocoo()
            all_rows.append(coo.row + r_off)
            all_cols.append(coo.col + c_off)
            all_data.append(coo.data)

        all_rows = np.concatenate(all_rows)
        all_cols = np.concatenate(all_cols)
        all_data = np.concatenate(all_data)

        H = sparse.csr_matrix((all_data, (all_rows, all_cols)), shape=(n_sol, n_sol))
        return H

    def _interface_displacements(self, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract relative interface displacement (dux, duy) from U."""
        Nv = self.Nv
        dux = U[self._intf1_ox:self._intf1_ox + Nv] - U[self._intf2_ox:self._intf2_ox + Nv]
        duy = U[self._intf1_oy:self._intf1_oy + Nv] - U[self._intf2_oy:self._intf2_oy + Nv]
        return dux, duy

    def _interface_phases(self, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute stacking phases at the interface for given U."""
        dux, duy = self._interface_displacements(U)
        Mu = self._Mu1
        dv = -(Mu[0, 0] * dux + Mu[0, 1] * duy)
        dw = -(Mu[1, 0] * dux + Mu[1, 1] * duy)
        return self._v0 + dv, self._w0 + dw

    def _scatter_interface_gradient(self, grad: np.ndarray, fx: np.ndarray, fy: np.ndarray):
        """Add GSFE force contributions to gradient at interface layer DOFs.

        fx, fy are forces on (dux, duy). Since dux = ux1 - ux2, the force
        on ux1 is +fx and on ux2 is -fx.
        """
        Nv = self.Nv
        grad[self._intf1_ox:self._intf1_ox + Nv] += fx
        grad[self._intf2_ox:self._intf2_ox + Nv] -= fx
        grad[self._intf1_oy:self._intf1_oy + Nv] += fy
        grad[self._intf2_oy:self._intf2_oy + Nv] -= fy

    def __call__(self, U: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute total energy and gradient."""
        self.eval_count += 1
        n_sol = self.conv.n_sol

        # --- Elastic energy (quadratic: E = 0.5 * U^T H U, grad = H U) ---
        HU = self._H_elastic @ U
        E_elastic = 0.5 * U.dot(HU)
        grad = HU.copy()

        # --- Interface GSFE energy ---
        v_int, w_int = self._interface_phases(U)
        V_int = self.gsfe_interface(v_int, w_int)
        V_min = self.gsfe_interface.minimum_value
        E_gsfe = float(np.sum((V_int - V_min) * self._area_v_norm))

        # GSFE gradient
        dVdv = self.gsfe_interface.dv(v_int, w_int)
        dVdw = self.gsfe_interface.dw(v_int, w_int)
        Mu = self._Mu1

        gsfe_force_x = -(Mu[0, 0] * dVdv + Mu[1, 0] * dVdw) * self._area_v_norm
        gsfe_force_y = -(Mu[0, 1] * dVdv + Mu[1, 1] * dVdw) * self._area_v_norm
        self._scatter_interface_gradient(grad, gsfe_force_x, gsfe_force_y)

        return E_elastic + E_gsfe, grad

    def hessp(self, U: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Compute Hessian-vector product H(U) @ p."""
        Nv = self.Nv

        # Elastic part (constant sparse matrix-vector product)
        result = self._H_elastic @ p

        # GSFE part (vertex-diagonal, fast)
        v_int, w_int = self._interface_phases(U)
        d2Vdv2 = self.gsfe_interface.d2v2(v_int, w_int)
        d2Vdw2 = self.gsfe_interface.d2w2(v_int, w_int)
        d2Vdvw = self.gsfe_interface.d2vw(v_int, w_int)
        Mu = self._Mu1
        w_v = self._area_v_norm

        # Extract interface displacement perturbation from p (direct slicing)
        dpx = p[self._intf1_ox:self._intf1_ox + Nv] - p[self._intf2_ox:self._intf2_ox + Nv]
        dpy = p[self._intf1_oy:self._intf1_oy + Nv] - p[self._intf2_oy:self._intf2_oy + Nv]

        # Phase perturbation
        dp_v = -(Mu[0, 0] * dpx + Mu[0, 1] * dpy)
        dp_w = -(Mu[1, 0] * dpx + Mu[1, 1] * dpy)

        # d^2V/d(phase)^2 @ d(phase)
        hv = d2Vdv2 * dp_v + d2Vdvw * dp_w
        hw = d2Vdvw * dp_v + d2Vdw2 * dp_w

        # Map back through chain rule to forces on (dux, duy)
        hfx = -(Mu[0, 0] * hv + Mu[1, 0] * hw) * w_v
        hfy = -(Mu[0, 1] * hv + Mu[1, 1] * hw) * w_v

        # Scatter: dux = ux1 - ux2, so force on ux1 is +hfx, on ux2 is -hfx
        self._scatter_interface_gradient(result, hfx, hfy)
        return result

    def hessian(self, U: np.ndarray) -> sparse.csr_matrix:
        """Compute the full Hessian as a sparse matrix.

        H = H_elastic (constant) + H_gsfe(U) (vertex-diagonal blocks).
        """
        Nv = self.Nv
        n_sol = self.conv.n_sol

        v_int, w_int = self._interface_phases(U)
        d2Vdv2 = self.gsfe_interface.d2v2(v_int, w_int)
        d2Vdw2 = self.gsfe_interface.d2w2(v_int, w_int)
        d2Vdvw = self.gsfe_interface.d2vw(v_int, w_int)
        Mu = self._Mu1
        w_v = self._area_v_norm

        # 2x2 GSFE Hessian blocks in (dux, duy) space per vertex
        H_dxdx = w_v * (Mu[0, 0] ** 2 * d2Vdv2 + 2 * Mu[0, 0] * Mu[1, 0] * d2Vdvw + Mu[1, 0] ** 2 * d2Vdw2)
        H_dxdy = w_v * (Mu[0, 0] * Mu[0, 1] * d2Vdv2 + (Mu[0, 0] * Mu[1, 1] + Mu[0, 1] * Mu[1, 0]) * d2Vdvw + Mu[1, 0] * Mu[1, 1] * d2Vdw2)
        H_dydy = w_v * (Mu[0, 1] ** 2 * d2Vdv2 + 2 * Mu[0, 1] * Mu[1, 1] * d2Vdvw + Mu[1, 1] ** 2 * d2Vdw2)

        # Build sparse GSFE Hessian by placing diagonal blocks at interface DOF positions
        # dux = U[intf1_ox:...] - U[intf2_ox:...], so the Hessian has entries at
        # (intf1, intf1), (intf1, intf2), (intf2, intf1), (intf2, intf2) with appropriate signs
        idx = np.arange(Nv)
        offsets = [
            (self._intf1_ox, self._intf1_ox, +1, +1),  # d^2/d(ux1)d(ux1)
            (self._intf1_ox, self._intf2_ox, +1, -1),  # d^2/d(ux1)d(ux2)
            (self._intf2_ox, self._intf1_ox, -1, +1),
            (self._intf2_ox, self._intf2_ox, -1, -1),
        ]

        all_rows = []
        all_cols = []
        all_data = []

        for r_off, c_off, sr, sc in offsets:
            # xx block
            all_rows.append(idx + r_off)
            all_cols.append(idx + c_off)
            all_data.append(sr * sc * H_dxdx)

        y_offsets = [
            (self._intf1_oy, self._intf1_oy, +1, +1),
            (self._intf1_oy, self._intf2_oy, +1, -1),
            (self._intf2_oy, self._intf1_oy, -1, +1),
            (self._intf2_oy, self._intf2_oy, -1, -1),
        ]
        for r_off, c_off, sr, sc in y_offsets:
            # yy block
            all_rows.append(idx + r_off)
            all_cols.append(idx + c_off)
            all_data.append(sr * sc * H_dydy)

        # Cross terms (xy and yx)
        xy_offsets = [
            (self._intf1_ox, self._intf1_oy, +1, +1),
            (self._intf1_ox, self._intf2_oy, +1, -1),
            (self._intf2_ox, self._intf1_oy, -1, +1),
            (self._intf2_ox, self._intf2_oy, -1, -1),
        ]
        for r_off, c_off, sr, sc in xy_offsets:
            all_rows.append(idx + r_off)
            all_cols.append(idx + c_off)
            all_data.append(sr * sc * H_dxdy)
            # Symmetric: yx = xy^T
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

        # Interface GSFE
        v_int, w_int = self._interface_phases(U)
        V_int = self.gsfe_interface(v_int, w_int)
        V_min = self.gsfe_interface.minimum_value
        gsfe_map = (V_int - V_min) / self.geometry.lattice.unit_cell_area

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
                e_el = 0.5 * K * trace**2 + 0.5 * G * ((exx - eyy) ** 2 + (exy + eyx) ** 2)
                maps_k[k] = (t2v @ e_el) / self.geometry.lattice.unit_cell_area

            elastic_maps[label] = maps_k

        return {"gsfe_interface": gsfe_map, **elastic_maps}
