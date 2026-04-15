"""High-level solver API for moire relaxation.

Typical usage::

    from moire_metrology import RelaxationSolver
    from moire_metrology.interfaces import GRAPHENE_GRAPHENE

    solver = RelaxationSolver()
    result = solver.solve(
        moire_interface=GRAPHENE_GRAPHENE,
        theta_twist=2.0,
    )
    result.plot_stacking()
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import spsolve

from .discretization import Discretization
from .energy import RelaxationEnergy
from .gsfe import GSFESurface
from .interfaces import Interface
from .lattice import HexagonalLattice, MoireGeometry
from .mesh import MoireMesh
from .result import RelaxationResult

if TYPE_CHECKING:
    from .discretization import PinnedConstraints


@dataclass
class SolverConfig:
    """Configuration for the relaxation solver.

    Parameters
    ----------
    method : str
        Optimization method.

        - ``'newton'`` (default) -- Newton's method with sparse direct Hessian
          solve. Fast convergence using precomputed elastic Hessian +
          vertex-diagonal GSFE Hessian. Can stall on multi-layer systems at
          low twist (Hessian becomes nearly indefinite); use
          ``'pseudo_dynamics'`` for those cases.
        - ``'pseudo_dynamics'`` -- implicit theta-method on the gradient flow
          dU/dt = -nabla E. At each step solves (M + beta*dt*H)*dU = -dt*grad
          with adaptive dt and energy-monitored step rejection. More robust
          than Newton on stiff/multi-layer systems. Ports the algorithm used
          in the paper MATLAB code.
        - ``'L-BFGS-B'`` -- Gradient-only quasi-Newton via
          scipy.optimize.minimize. Slower convergence but no Hessian needed.
    max_iter : int
        Maximum number of optimizer iterations.
    gtol : float
        Absolute gradient norm tolerance for convergence. Convergence is
        declared when ``|grad| < gtol``. For L-BFGS-B this is passed to
        scipy as the ``gtol`` option (max-norm threshold).
    rtol : float
        Relative gradient tolerance. Convergence is declared when
        ``|grad| / |grad_initial| < rtol``, i.e. the gradient has dropped
        by this factor from its initial value. This criterion matters at
        low twist angles where the absolute gradient norm at the energy
        minimum can be O(1-10) due to large total energies, making any
        reasonable absolute gtol unreachable. For L-BFGS-B the check is
        applied post-hoc after scipy returns.
    etol : float
        Energy stagnation tolerance. If the fractional energy improvement
        over the last ``etol_window`` accepted steps falls below ``etol``,
        convergence is declared (newton and pseudo_dynamics only).
    etol_window : int
        Number of recent iterations over which to measure energy stagnation.
    pixel_size : float
        Target mesh element size in nm.
    n_scale : int
        Number of moire unit cells in each direction.
    display : bool
        If True, print progress during optimization.
    min_mesh_points : int
        Floor on the mesh grid resolution per direction. Default 100.
    beta : float
        Theta-method parameter for the 'pseudo_dynamics' solver. 0 = explicit
        Euler (unstable), 1 = fully implicit backward Euler (default,
        unconditionally stable). Values in [0.5, 1.0] are stable.
    dt0 : float or None
        Initial pseudo-time step for the 'pseudo_dynamics' solver. If None,
        chosen automatically as ~1/(K1+G1) so that dt0·\|H\| ~ 1.
    linear_solver : str
        Linear-system solver used inside each pseudo_dynamics step.

        - ``'direct'`` (default) -- build the sparse Hessian explicitly and
          call spsolve. Per-iteration cost is the LU factorization, which
          scales like O(n_sol^1.5) on 2D meshes. Best for small/moderate
          problems.
        - ``'iterative'`` -- matrix-free preconditioned MINRES, using only
          Hessian-vector products and a Jacobi preconditioner from the
          diagonal of the constant elastic Hessian. Much faster per iteration
          on large problems (n_sol >> 10^4). Uses MINRES (not CG) so it stays
          robust when the operator is symmetric but indefinite.
    linear_solver_tol : float
        Relative-residual tolerance for the iterative linear solver. Only
        used when linear_solver='iterative'. Default 1e-6.
    linear_solver_maxiter : int
        Max iterations for the iterative linear solver per pseudo_dynamics
        step. Default 200. The outer pseudo_dynamics loop will rebuild the
        operator and try again with a smaller dt if MINRES fails to converge.
    """

    method: str = "newton"
    max_iter: int = 200
    gtol: float = 1e-6
    rtol: float = 1e-4
    etol: float = 1e-6
    etol_window: int = 5
    pixel_size: float = 0.2
    n_scale: int = 1
    display: bool = True
    min_mesh_points: int = 100
    beta: float = 1.0
    dt0: float | None = None
    linear_solver: str = "direct"
    linear_solver_tol: float = 1e-6
    linear_solver_maxiter: int = 200
    elastic_strain: str = "cauchy"


def _newton_solve(energy_func: RelaxationEnergy, U0: np.ndarray,
                  max_iter: int, gtol: float, rtol: float,
                  etol: float, etol_window: int,
                  display: bool,
                  linear_solver: str = "direct",
                  linear_solver_tol: float = 1e-6,
                  linear_solver_maxiter: int = 500) -> dict:
    """Damped Newton solver with sparse direct Hessian factorization.

    Uses Levenberg-Marquardt damping to handle indefinite Hessians:
        (H + mu*I) @ dU = -grad

    The damping parameter mu is adapted:
    - Decreased when full Newton steps succeed (energy decreases)
    - Increased when the Hessian is indefinite or steps fail

    Convergence is declared when ANY of:
        |grad|        < gtol                     (absolute), OR
        |grad|/|grad0| < gtol                    (relative to initial), OR
        (E_best - E_new) / |E_best| < etol       (energy stagnation over
                                                  ``etol_window`` iters).
    The relative criterion matters at low twist angles where the absolute
    gradient norm at the energy minimum can be O(1-10) due to large total
    energies, making any reasonable absolute gtol unreachable. The
    energy-stagnation criterion is the fallback when neither gradient
    criterion can be satisfied because Newton is making consistent but
    tiny progress on a shallow minimum (common in very large multi-layer
    systems where each Newton step is expensive).

    A KeyboardInterrupt during the iteration returns the best iterate
    seen so far (by energy), so that a long-running solve can be stopped
    with Ctrl-C and still yield a usable result for plotting/debugging.

    The elastic Hessian is constant (precomputed). Only the GSFE Hessian
    changes each iteration. The linear system is solved with sparse LU.
    """
    from scipy.sparse import eye as speye
    from scipy.sparse.linalg import cg as sparse_cg

    U = U0.copy()
    E, grad = energy_func(U)
    gnorm = np.linalg.norm(grad)
    gnorm0 = max(gnorm, 1.0)
    n_sol = len(U)
    nit = 0
    nfev = 1
    t_start = perf_counter()

    # Track best iterate for KeyboardInterrupt recovery.
    U_best, E_best, grad_best = U.copy(), E, grad.copy()

    recent_E: list[float] = [E]

    def converged(gn: float) -> bool:
        return gn < gtol or (gn / gnorm0) < rtol

    def stagnated() -> bool:
        if len(recent_E) < etol_window + 1:
            return False
        window = recent_E[-(etol_window + 1):]
        delta = window[0] - window[-1]
        scale = max(abs(window[-1]), 1.0)
        return (delta / scale) < etol

    mu = 1e-4 * (energy_func.K1 + energy_func.G1)
    mu_min = 1e-10
    mu_max = 1e10
    I_sp = speye(n_sol, format="csr")

    interrupted = False
    exit_reason: str | None = None
    try:
        for nit in range(1, max_iter + 1):
            if converged(gnorm):
                if gnorm < gtol:
                    exit_reason = (
                        f"converged (absolute |grad| = {gnorm:.2e} < gtol = {gtol:.0e})")
                else:
                    exit_reason = (
                        f"converged (relative |grad|/|grad0| = "
                        f"{gnorm / gnorm0:.2e} < rtol = {rtol:.0e})")
                break
            if stagnated():
                window = recent_E[-(etol_window + 1):]
                de = (window[0] - window[-1]) / max(abs(window[-1]), 1.0)
                exit_reason = (
                    f"converged (energy stagnation dE/E = {de:.2e} "
                    f"< etol = {etol:.0e} over {etol_window} iters)")
                break

            H = energy_func.hessian(U, modified=True)
            H_damped = H + mu * I_sp

            # Solve H_damped @ dU = -grad.
            # - "direct": sparse LU (exact but fill-in is bad for
            #   large multi-layer stacks).
            # - "iterative": preconditioned CG. The modified Hessian
            #   is PD by construction (PR #16), so CG is valid. A
            #   diagonal (Jacobi) preconditioner is cheap and gives
            #   decent clustering of the spectrum for these systems.
            try:
                if linear_solver == "direct":
                    dU = spsolve(H_damped, -grad)
                else:
                    diag = H_damped.diagonal()
                    diag = np.where(np.abs(diag) > 0, diag, 1.0)
                    M_inv = sparse.diags(1.0 / diag, format="csr")
                    dU, info = sparse_cg(
                        H_damped, -grad,
                        rtol=linear_solver_tol,
                        maxiter=linear_solver_maxiter,
                        M=M_inv,
                    )
                    if info != 0 and display:
                        print(f"  [iter {nit}] CG did not fully converge "
                              f"(info={info}); using best iterate.")
            except Exception:
                mu *= 10
                continue

            slope = grad.dot(dU)
            if slope > 0:
                mu *= 10
                mu = min(mu, mu_max)
                continue

            U_trial = U + dU
            E_new, grad_new = energy_func(U_trial)
            nfev += 1

            if E_new < E:
                actual_reduction = E - E_new
                predicted_reduction = -slope - 0.5 * dU.dot(H @ dU)
                rho = actual_reduction / max(abs(predicted_reduction), 1e-30)

                U = U_trial
                E = E_new
                grad = grad_new
                gnorm = np.linalg.norm(grad)
                recent_E.append(E)

                if E < E_best:
                    U_best, E_best, grad_best = U.copy(), E, grad.copy()

                if rho > 0.75:
                    mu = max(mu / 3, mu_min)
                elif rho < 0.25:
                    mu *= 2
            else:
                mu *= 4
                mu = min(mu, mu_max)

            if display and (nit % 5 == 0 or nit <= 3):
                elapsed = perf_counter() - t_start
                rel = gnorm / gnorm0
                print(f"  iter {nit:4d}: E = {E:.4f}, |grad| = {gnorm:.2e} "
                      f"(rel {rel:.2e}), mu = {mu:.2e}, t = {elapsed:.1f}s")
    except KeyboardInterrupt:
        interrupted = True
        exit_reason = "interrupted (returning best iterate)"
        if display:
            print(f"\n  KeyboardInterrupt at iter {nit}; returning best iterate "
                  f"(E = {E_best:.4f}).")

    if exit_reason is None:
        exit_reason = f"max iterations ({max_iter}) reached"

    success = (not interrupted) and exit_reason is not None and (
        "converged" in exit_reason
    )
    return {
        "x": U_best, "fun": E_best, "jac": grad_best, "nit": nit, "nfev": nfev,
        "message": exit_reason, "success": success,
    }


def _pseudo_dynamics_solve(energy_func: RelaxationEnergy, U0: np.ndarray,
                           max_iter: int, gtol: float, rtol: float,
                           beta: float,
                           dt0: float | None, display: bool,
                           linear_solver: str = "direct",
                           linear_solver_tol: float = 1e-6,
                           linear_solver_maxiter: int = 200) -> dict:
    """Implicit theta-method pseudo-time-stepping relaxation solver.

    Solves the gradient flow ODE  M·dU/dt = -∇E(U)  by a generalized
    theta-method. Linearizing ∇E around U_prev gives the per-step linear
    system

        (M + β·dt·H(U_prev)) · ΔU = -dt · ∇E(U_prev)

    where M is the mass matrix (taken as identity here, matching the
    Amat0 = I structure of the paper MATLAB code on a periodic mesh),
    H is the analytic Hessian, and β ∈ [0, 1] is the implicit-explicit
    blend (β=1 is fully implicit/backward-Euler, β=0.5 is Crank-Nicolson).

    The pseudo-time step dt is adapted at each iteration:
      - On energy increase: reject the step, halve dt
      - On small relative ΔU (< 1%): accept and grow dt by 5%
      - On large relative ΔU: accept and shrink dt by 5%

    Convergence is declared when the gradient norm (absolute OR relative
    to initial) stays below gtol for several consecutive accepted steps.

    Compared to the Newton solver in this file:
      - The mass matrix M (positive definite) regularises the linear
        system, so we don't need Levenberg-Marquardt damping or worry
        about indefinite Hessians.
      - dt acts as an implicit trust radius — small dt → close to
        gradient descent, large dt → close to a Newton step.
      - The energy-monitored rewind avoids the LM stall mode where the
        damping pins high and Newton can't escape a saddle/plateau.

    Linear-solve modes (selected via the `linear_solver` argument):
      - 'direct' (default): build the full sparse Hessian and call spsolve.
        Per-iteration cost is the LU factorization, scaling like O(n^1.5)
        on 2D meshes. Best for small/moderate problems.
      - 'iterative': matrix-free preconditioned MINRES, using only
        Hessian-vector products via energy_func.hessp(U, p). The Jacobi
        preconditioner is built once from the diagonal of the constant
        elastic Hessian and reused at every step. MINRES (not CG) so the
        method stays robust when β·dt·H is symmetric but indefinite.
        Much faster than 'direct' on large problems (n_sol >> 10^4).

    This solver ports the algorithm of
    docs_internal/relaxation_code_2D_solver_periodic_ver6.m, simplified
    for the periodic-mesh case (no normal/tangent edge BC machinery,
    Amat0 = identity).
    """
    from scipy.sparse import eye as speye
    from scipy.sparse.linalg import LinearOperator, minres

    U = U0.copy()
    E, grad = energy_func(U)
    gnorm = np.linalg.norm(grad)
    gnorm0 = max(gnorm, 1.0)
    n_sol = len(U)
    nfev = 1
    nit = 0
    n_accepted = 0
    n_rejected = 0
    t_start = perf_counter()

    # Convergence: same OR-criterion as the Newton solver.
    def converged(gn: float) -> bool:
        return gn < gtol or (gn / gnorm0) < rtol

    # Mass matrix: identity (matches the periodic-mesh case in the
    # MATLAB code where Amat0 has 1's on the interior diagonal). A
    # vertex-area-weighted lumped mass would be more physical for
    # gradient flow on the L^2 metric, but identity is simpler and
    # is what the published algorithm uses.
    M = speye(n_sol, format="csr")

    # Initial pseudo-time step. If not specified, choose so that
    # dt0 · |H_elastic_diag| ~ 1, i.e. dt0 ~ 1/(K1+G1).
    if dt0 is None:
        dt = 1.0 / max(energy_func.K1 + energy_func.G1, 1.0)
    else:
        dt = dt0

    # ---- Iterative-solver bookkeeping ------------------------------
    # The constant elastic Hessian gives us a free Jacobi preconditioner.
    # When constraints are present, we need the diagonal in the free-DOF
    # space, not the full DOF space. Cache it once.
    H_elastic_diag_full = None
    H_elastic_diag_free = None
    if linear_solver == "iterative":
        H_elastic_diag_full = np.asarray(
            energy_func._H_elastic.diagonal()
        ).ravel()
        if energy_func.constraints is not None:
            H_elastic_diag_free = H_elastic_diag_full[
                energy_func.constraints.free_indices
            ]
        else:
            H_elastic_diag_free = H_elastic_diag_full
        # Sanity: positive (the elastic Hessian is PD on the free DOFs)
        H_elastic_diag_free = np.maximum(H_elastic_diag_free, 0.0)
    elif linear_solver != "direct":
        raise ValueError(
            f"Unknown linear_solver={linear_solver!r}; "
            f"expected 'direct' or 'iterative'."
        )

    # Convergence requires the gradient norm to be below tolerance for
    # this many consecutive accepted steps. Avoids early exit on noise.
    count_conv_required = 3
    count_conv = 0

    # Step size adaptation thresholds.
    change_fac_small = 1e-2  # if relative ΔU below this, dt grows
    dt_grow = 1.05
    dt_shrink = 0.95
    dt_min = 1e-15
    dt_max = 1e6

    # Tolerance on energy increase: small fraction of |E|.
    dE_rel_tol = 1e-8

    # Hard cap on consecutive rejections so we never spin forever.
    max_consec_rejects = 20

    consec_rejects = 0

    for nit in range(1, max_iter + 1):
        if count_conv >= count_conv_required:
            break

        rhs = -dt * grad

        # Solve (M + β·dt·H(U_prev)) · ΔU = rhs
        # The system is well-posed because M is positive definite even
        # when H is indefinite or near-singular.
        try:
            if linear_solver == "direct":
                # Build the implicit Hessian explicitly and call spsolve.
                H = energy_func.hessian(U)
                A = M + (beta * dt) * H
                dU = spsolve(A, rhs)
            else:
                # Matrix-free MINRES against energy_func.hessp.
                # Captures U and dt at the current step (rebound each iter).
                _U_local = U
                _coef = beta * dt

                def _matvec(p):
                    return p + _coef * energy_func.hessp(_U_local, p)

                A_op = LinearOperator(
                    (n_sol, n_sol), matvec=_matvec, dtype=np.float64,
                )

                # Jacobi preconditioner: (1 + β·dt·diag(H_elastic))^{-1}.
                # Approximates the system diagonal cheaply; the GSFE
                # contribution to the diagonal is small relative to the
                # elastic contribution and is left out.
                _precond = 1.0 / (1.0 + _coef * H_elastic_diag_free)
                M_op = LinearOperator(
                    (n_sol, n_sol),
                    matvec=lambda p: _precond * p,
                    dtype=np.float64,
                )

                dU, info = minres(
                    A_op, rhs, M=M_op,
                    rtol=linear_solver_tol,
                    maxiter=linear_solver_maxiter,
                )
                if info < 0:
                    # MINRES reports illegal input — treat as a numerical
                    # breakdown and let the outer rejection logic shrink dt.
                    raise RuntimeError(f"minres returned info={info}")
                # info > 0 means did-not-converge; we still use dU as a
                # best-effort step and let the energy check accept/reject.
        except Exception:
            # Numerical breakdown — shrink dt and retry.
            dt *= 0.5
            consec_rejects += 1
            if consec_rejects > max_consec_rejects or dt < dt_min:
                break
            continue

        U_trial = U + dU
        E_new, grad_new = energy_func(U_trial)
        nfev += 1
        dE = E_new - E

        if dE > dE_rel_tol * max(abs(E), 1.0):
            # Reject: energy increased. Halve dt and try again from U_prev.
            dt = max(dt * 0.5, dt_min)
            n_rejected += 1
            consec_rejects += 1
            if consec_rejects > max_consec_rejects:
                # Stuck — give up rather than spin.
                break
            continue

        # Accept the step.
        consec_rejects = 0
        n_accepted += 1
        # Relative-displacement diagnostic for dt adaptation.
        denom = np.maximum(np.abs(U_trial), 1e-12).max()
        change_fac_max = float(np.abs(dU).max() / denom)

        U = U_trial
        E = E_new
        grad = grad_new
        gnorm = np.linalg.norm(grad)

        # Adapt dt based on how big the accepted step was relative to U.
        if change_fac_max < change_fac_small:
            dt = min(dt * dt_grow, dt_max)
        else:
            dt = max(dt * dt_shrink, dt_min)

        # Convergence accounting (require several consecutive good steps).
        if converged(gnorm):
            count_conv += 1
        else:
            count_conv = 0

        if display and (nit % 10 == 0 or nit <= 3):
            elapsed = perf_counter() - t_start
            rel = gnorm / gnorm0
            print(f"  iter {nit:4d}: E = {E:.4f}, |grad| = {gnorm:.2e} "
                  f"(rel {rel:.2e}), dt = {dt:.2e}, t = {elapsed:.1f}s")

        if count_conv >= count_conv_required:
            break

    success = converged(gnorm) and count_conv >= count_conv_required
    if success:
        if gnorm < gtol:
            message = (
                f"converged (absolute |grad| = {gnorm:.2e} < gtol = {gtol:.0e})")
        else:
            message = (
                f"converged (relative |grad|/|grad0| = "
                f"{gnorm / gnorm0:.2e} < rtol = {rtol:.0e})")
    elif nit >= max_iter:
        message = f"max iterations ({max_iter}) reached"
    else:
        message = "stalled (consecutive rejections exceeded)"

    if display:
        print(f"  pseudo_dynamics: {n_accepted} accepted, {n_rejected} rejected steps")

    return {
        "x": U, "fun": E, "jac": grad, "nit": nit, "nfev": nfev,
        "message": message, "success": success,
    }


_LEGACY_KWARGS = {"material1", "material2", "nlayer1", "nlayer2"}


def _raise_legacy_kwargs(kwargs: dict) -> None:
    """Friendly redirect for code calling the v0.1.0 solve() signature."""
    leftover = set(kwargs) - _LEGACY_KWARGS
    if leftover:
        # Truly unknown kwargs — let Python raise its normal TypeError.
        raise TypeError(
            f"RelaxationSolver.solve() got unexpected keyword argument(s): {sorted(leftover)}"
        )
    raise TypeError(
        "RelaxationSolver.solve() no longer takes material1/material2/nlayer1/nlayer2 "
        "in v0.2.0. GSFE has moved off Material and onto Interface. Replace:\n"
        "    solver.solve(material1=GRAPHENE, material2=GRAPHENE, theta_twist=1.05)\n"
        "with:\n"
        "    from moire_metrology.interfaces import GRAPHENE_GRAPHENE\n"
        "    solver.solve(moire_interface=GRAPHENE_GRAPHENE, theta_twist=1.05)\n"
        "For multi-layer flakes, also pass top_interface= and/or bottom_interface= "
        "and use n_top= / n_bottom= instead of nlayer1= / nlayer2=. See CHANGELOG.md "
        "for the full migration."
    )


def _validate_flake_interfaces(
    moire_interface: Interface,
    top_interface: Interface | None,
    bottom_interface: Interface | None,
    n_top: int,
    n_bottom: int,
) -> None:
    """Sanity-check that the supplied interfaces are coherent with the flake sizes.

    Catches common mistakes at the API boundary instead of producing a
    confusing error 200 lines into the energy assembler.
    """
    if n_top > 1:
        if top_interface is None:
            raise ValueError(
                f"n_top={n_top} requires a `top_interface=` (the homobilayer "
                f"interface inside the top flake). For a {moire_interface.top.name} "
                f"top flake, pass the matching homobilayer interface from "
                f"moire_metrology.interfaces."
            )
        if not top_interface.is_homobilayer:
            raise ValueError(
                "top_interface must be a homobilayer interface "
                "(top_interface.bottom is top_interface.top), "
                f"got {top_interface.name} which is heterogeneous."
            )
        if top_interface.top is not moire_interface.top:
            raise ValueError(
                "top_interface material does not match the top flake material. "
                f"moire_interface.top={moire_interface.top.name!r}, "
                f"top_interface.top={top_interface.top.name!r}."
            )

    if n_bottom > 1:
        if bottom_interface is None:
            raise ValueError(
                f"n_bottom={n_bottom} requires a `bottom_interface=` (the homobilayer "
                f"interface inside the bottom flake). For a {moire_interface.bottom.name} "
                f"bottom flake, pass the matching homobilayer interface from "
                f"moire_metrology.interfaces."
            )
        if not bottom_interface.is_homobilayer:
            raise ValueError(
                "bottom_interface must be a homobilayer interface "
                "(bottom_interface.bottom is bottom_interface.top), "
                f"got {bottom_interface.name} which is heterogeneous."
            )
        if bottom_interface.bottom is not moire_interface.bottom:
            raise ValueError(
                "bottom_interface material does not match the bottom flake material. "
                f"moire_interface.bottom={moire_interface.bottom.name!r}, "
                f"bottom_interface.bottom={bottom_interface.bottom.name!r}."
            )


class RelaxationSolver:
    """Solver for atomic relaxation in twisted 2D heterostructures."""

    def __init__(self, config: SolverConfig | None = None):
        self.config = config or SolverConfig()

    def solve(
        self,
        moire_interface: Interface | None = None,
        *,
        top_interface: Interface | None = None,
        bottom_interface: Interface | None = None,
        n_top: int = 1,
        n_bottom: int = 1,
        theta_twist: float = 0.0,
        delta: float | None = None,
        theta0: float = 0.0,
        initial_solution: np.ndarray | None = None,
        constraints: "PinnedConstraints | None" = None,
        fix_top: bool = False,
        fix_bottom: bool = False,
        pin_mean: bool = False,
        mesh: MoireMesh | None = None,
        **legacy_kwargs,
    ) -> RelaxationResult:
        """Solve the relaxation problem.

        Parameters
        ----------
        moire_interface : Interface
            The twisted A-B interface between the bottom layer of the top
            flake and the top layer of the bottom flake. Carries both
            materials (``moire_interface.bottom`` is the bottom-flake
            material, ``moire_interface.top`` is the top-flake material)
            as well as the GSFE coefficients for their stacking.
        top_interface : Interface, optional
            Required iff ``n_top > 1``. The homobilayer interface used
            between successive layers within the top flake. Must have
            ``bottom == top == moire_interface.top``.
        bottom_interface : Interface, optional
            Required iff ``n_bottom > 1``. The homobilayer interface used
            between successive layers within the bottom flake. Must have
            ``bottom == top == moire_interface.bottom``.
        n_top, n_bottom : int
            Number of layers in each flake.
        theta_twist : float
            Twist angle in degrees.
        delta : float or None
            Lattice mismatch. If None, computed from the two materials.
        theta0 : float
            Lattice orientation angle in degrees.
        initial_solution : ndarray or None
            Initial guess. If None, starts from zero.
        constraints : PinnedConstraints or None
            If set, pins certain DOFs to fixed displacements while
            optimizing the rest. Build via PinningMap.build_constraints().
            Mutually exclusive with fix_top / fix_bottom.
        fix_top : bool
            Pin all DOFs of the topmost layer (top of the top flake) to
            zero. Use this to clamp the upper free surface of the
            heterostructure and approximate a semi-infinite top.
        fix_bottom : bool
            Pin all DOFs of the bottommost layer (bottom of the bottom
            flake) to zero. Use this to clamp the substrate and
            approximate a semi-infinite bottom — typical for simulating
            a twisted flake on a thick substrate (e.g. graphene on
            graphite).
        mesh : MoireMesh or None
            Pre-built mesh to use. If None (default), the solver builds a
            periodic moire-cell mesh from the SolverConfig parameters
            (pixel_size, n_scale, min_mesh_points). Pass a mesh built via
            ``moire_metrology.mesh.generate_finite_mesh`` (or any other
            MoireMesh constructor) to relax on a finite, non-periodic
            domain — typically combined with ``constraints`` from
            ``PinningMap.build_constraints`` to pin selected stacking
            sites in the experimental image.

        Notes
        -----
        Internally the solver still uses the legacy "stack 1" / "stack 2"
        numbering, where stack 1 is the *top* flake and stack 2 is the
        *bottom* flake. This is an implementation detail of the
        construction layer; the public API is in terms of
        ``moire_interface.top`` / ``moire_interface.bottom`` and the
        ``n_top`` / ``n_bottom`` flake sizes. The translation happens
        once at the top of this method.
        """
        if legacy_kwargs:
            _raise_legacy_kwargs(legacy_kwargs)
        if moire_interface is None:
            raise TypeError(
                "RelaxationSolver.solve() requires a `moire_interface` argument. "
                "Pass a bundled Interface from moire_metrology.interfaces (e.g. "
                "GRAPHENE_GRAPHENE) or construct your own Interface."
            )

        _validate_flake_interfaces(
            moire_interface, top_interface, bottom_interface, n_top, n_bottom
        )

        # Translate from the public (top/bottom) vocabulary to the internal
        # (stack-1 = top flake, stack-2 = bottom flake) numbering. The
        # solver guts and the npz schema still use stack 1 / stack 2.
        material1 = moire_interface.top      # stack 1 = top flake
        material2 = moire_interface.bottom   # stack 2 = bottom flake
        nlayer1 = n_top
        nlayer2 = n_bottom

        cfg = self.config

        if delta is None:
            delta = material1.lattice_constant / material2.lattice_constant - 1.0

        lattice = HexagonalLattice(alpha=material2.lattice_constant, theta0=theta0)
        geometry = MoireGeometry(lattice, theta_twist=theta_twist, delta=delta)

        if cfg.display:
            print(f"Moire wavelength: {geometry.wavelength:.2f} nm")
            print(f"Twist angle: {theta_twist:.4f} deg, delta: {delta:.6f}")

        if mesh is None:
            mesh = MoireMesh.generate(
                geometry,
                pixel_size=cfg.pixel_size,
                n_scale=cfg.n_scale,
                min_points=cfg.min_mesh_points,
            )
        if cfg.display:
            kind = "periodic" if mesh.is_periodic else "finite"
            print(f"Mesh: {mesh.n_vertices} vertices, "
                  f"{mesh.n_triangles} triangles ({kind})")

        disc = Discretization(mesh, geometry)
        conv = disc.build_conversion_matrices(nlayer1=nlayer1, nlayer2=nlayer2)

        # Outer-layer clamps build a PinnedConstraints automatically.
        if (fix_top or fix_bottom or pin_mean):
            if constraints is not None:
                raise ValueError(
                    "fix_top / fix_bottom / pin_mean cannot be combined "
                    "with an explicit constraints argument; build a single "
                    "PinnedConstraints manually if you need both kinds of "
                    "pinning."
                )
            from .discretization import build_outer_layer_constraints
            constraints = build_outer_layer_constraints(
                conv, fix_top=fix_top, fix_bottom=fix_bottom,
                pin_mean=pin_mean,
            )

        # GSFE comes from the interfaces, not the materials. The moiré
        # interface drives the twisted flake-flake registry; the
        # homobilayer interfaces (when present) drive the intra-flake
        # registry of multi-layer flakes.
        gsfe_interface = GSFESurface(moire_interface.gsfe_coeffs)
        gsfe_flake1 = (
            GSFESurface(top_interface.gsfe_coeffs) if nlayer1 > 1 else None
        )
        gsfe_flake2 = (
            GSFESurface(bottom_interface.gsfe_coeffs) if nlayer2 > 1 else None
        )

        I1_vect = J1_vect = I2_vect = J2_vect = None
        if nlayer1 > 1 and top_interface is not None and top_interface.stacking_func is not None:
            I1_vect = np.array([top_interface.stacking_func(k)[0] for k in range(1, nlayer1)])
            J1_vect = np.array([top_interface.stacking_func(k)[1] for k in range(1, nlayer1)])
        if nlayer2 > 1 and bottom_interface is not None and bottom_interface.stacking_func is not None:
            I2_vect = np.array([bottom_interface.stacking_func(k)[0] for k in range(1, nlayer2)])
            J2_vect = np.array([bottom_interface.stacking_func(k)[1] for k in range(1, nlayer2)])

        if cfg.display:
            t_start = perf_counter()
            print("Building energy functional...")

        energy_func = RelaxationEnergy(
            disc=disc, conv=conv, geometry=geometry,
            gsfe_interface=gsfe_interface,
            K1=material1.bulk_modulus * nlayer1,
            G1=material1.shear_modulus * nlayer1,
            K2=material2.bulk_modulus * nlayer2,
            G2=material2.shear_modulus * nlayer2,
            nlayer1=nlayer1, nlayer2=nlayer2,
            gsfe_flake1=gsfe_flake1, gsfe_flake2=gsfe_flake2,
            I1_vect=I1_vect, J1_vect=J1_vect,
            I2_vect=I2_vect, J2_vect=J2_vect,
            constraints=constraints,
            elastic_strain=cfg.elastic_strain,
        )

        if cfg.display:
            print(f"  Done in {perf_counter() - t_start:.1f}s")
            if constraints is not None:
                print(f"  Constraints: {constraints.n_free} free / "
                      f"{len(constraints.pinned_indices)} pinned DOFs")

        # Initial guess — in free-DOF space if constrained
        if initial_solution is not None:
            U0 = initial_solution
        elif constraints is not None:
            U0 = np.zeros(constraints.n_free)
        else:
            U0 = np.zeros(conv.n_sol)
        E0, _ = energy_func(U0)
        if cfg.display:
            print(f"Unrelaxed energy: {E0:.2f} meV (total for domain)")
            print(f"Optimizing with {cfg.method}...")
            t_start = perf_counter()

        # Run optimizer
        if cfg.method in ("newton", "pseudo_dynamics"):
            if cfg.method == "newton":
                res = _newton_solve(
                    energy_func, U0, cfg.max_iter, cfg.gtol, cfg.rtol,
                    cfg.etol, cfg.etol_window, cfg.display,
                    linear_solver=cfg.linear_solver,
                    linear_solver_tol=cfg.linear_solver_tol,
                    linear_solver_maxiter=cfg.linear_solver_maxiter,
                )
            else:
                res = _pseudo_dynamics_solve(
                    energy_func, U0, cfg.max_iter, cfg.gtol, cfg.rtol,
                    beta=cfg.beta, dt0=cfg.dt0, display=cfg.display,
                    linear_solver=cfg.linear_solver,
                    linear_solver_tol=cfg.linear_solver_tol,
                    linear_solver_maxiter=cfg.linear_solver_maxiter,
                )

            class _Result:
                pass
            result = _Result()
            result.x = res["x"]
            result.fun = res["fun"]
            result.nit = res["nit"]
            result.nfev = res["nfev"]
            result.message = res["message"]
            result.success = res["success"]
        else:
            # Fallback to scipy.optimize.minimize (L-BFGS-B, etc.)
            # Capture initial gradient norm for post-hoc relative check.
            _, grad0 = energy_func(U0)
            gnorm0 = max(np.linalg.norm(grad0), 1.0)

            options = {"maxiter": cfg.max_iter, "gtol": cfg.gtol}
            if cfg.method == "L-BFGS-B":
                options["maxcor"] = 20
                options["maxls"] = 40
                options["ftol"] = 1e-15
            result = minimize(
                energy_func, U0, method=cfg.method, jac=True, options=options,
            )

            # Post-hoc relative convergence check: scipy's L-BFGS-B only
            # uses an absolute gtol.  If the relative criterion is met,
            # override success/message so the result honestly reports
            # convergence even when the absolute gtol was unreachable.
            if not result.success and hasattr(result, "jac"):
                gnorm_final = np.linalg.norm(result.jac)
                rel = gnorm_final / gnorm0
                if rel < cfg.rtol:
                    result.success = True
                    result.message = (
                        f"converged (relative |grad|/|grad0| = "
                        f"{rel:.2e} < rtol = {cfg.rtol:.0e})")

        if cfg.display:
            elapsed = perf_counter() - t_start
            print(f"Optimization finished: {result.message}")
            print(f"  Final energy: {result.fun:.4f} meV")
            print(f"  Iterations: {result.nit}, func evals: {result.nfev}, time: {elapsed:.1f}s")

        # Extract results — expand to full DOF space if constrained
        U_opt_raw = result.x
        if constraints is not None:
            U_opt = constraints.expand(U_opt_raw)
        else:
            U_opt = U_opt_raw
        Nv = mesh.n_vertices

        ux1 = np.zeros((nlayer1, Nv))
        uy1 = np.zeros((nlayer1, Nv))
        ux2 = np.zeros((nlayer2, Nv))
        uy2 = np.zeros((nlayer2, Nv))

        for k in range(nlayer1):
            ux1[k] = conv.conv_x1[k * Nv : (k + 1) * Nv] @ U_opt
            uy1[k] = conv.conv_y1[k * Nv : (k + 1) * Nv] @ U_opt
        for k in range(nlayer2):
            ux2[k] = conv.conv_x2[k * Nv : (k + 1) * Nv] @ U_opt
            uy2[k] = conv.conv_y2[k * Nv : (k + 1) * Nv] @ U_opt

        emaps = energy_func.energy_maps(U_opt)

        return RelaxationResult(
            mesh=mesh, geometry=geometry,
            moire_interface=moire_interface,
            top_interface=top_interface,
            bottom_interface=bottom_interface,
            displacement_x1=ux1, displacement_y1=uy1,
            displacement_x2=ux2, displacement_y2=uy2,
            total_energy=result.fun, unrelaxed_energy=E0,
            gsfe_map=emaps["gsfe_interface"],
            elastic_map1=emaps["elastic_1"],
            elastic_map2=emaps["elastic_2"],
            solution_vector=U_opt, optimizer_result=result,
        )
