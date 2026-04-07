"""High-level solver API for moire relaxation.

Typical usage:
    from moire_metrology import RelaxationSolver, GRAPHENE

    solver = RelaxationSolver()
    result = solver.solve(
        material1=GRAPHENE,
        material2=GRAPHENE,
        theta_twist=2.0,
    )
    result.plot_stacking()
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import spsolve

from .discretization import PeriodicDiscretization
from .energy import RelaxationEnergy
from .gsfe import GSFESurface
from .lattice import HexagonalLattice, MoireGeometry
from .materials import Material
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
        Optimization method:
        - 'newton' (default): Newton's method with sparse direct Hessian solve.
          Fast convergence using precomputed elastic Hessian + vertex-diagonal GSFE Hessian.
          Can stall on multi-layer systems at low twist (Hessian becomes nearly
          indefinite); use 'pseudo_dynamics' for those cases.
        - 'pseudo_dynamics': implicit theta-method on the gradient flow
          dU/dt = -∇E. At each step solves (M + β·dt·H)·ΔU = -dt·grad with
          adaptive dt and energy-monitored step rejection. More robust than
          Newton on stiff/multi-layer systems. Ports the algorithm used in
          the paper MATLAB code (relaxation_code_2D_solver_periodic_ver6.m).
        - 'L-BFGS-B': Gradient-only quasi-Newton via scipy.optimize.minimize.
          Slower convergence but no Hessian needed.
    max_iter : int
        Maximum number of optimizer iterations.
    gtol : float
        Gradient norm tolerance for convergence (used by 'newton' and
        'pseudo_dynamics'; both use absolute OR relative-to-initial criterion).
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
        chosen automatically as ~1/(K1+G1) so that dt0·|H| ~ 1.
    """

    method: str = "newton"
    max_iter: int = 200
    gtol: float = 1e-6
    pixel_size: float = 0.2
    n_scale: int = 1
    display: bool = True
    min_mesh_points: int = 100
    beta: float = 1.0
    dt0: float | None = None


def _newton_solve(energy_func: RelaxationEnergy, U0: np.ndarray,
                  max_iter: int, gtol: float, display: bool) -> dict:
    """Damped Newton solver with sparse direct Hessian factorization.

    Uses Levenberg-Marquardt damping to handle indefinite Hessians:
        (H + mu*I) @ dU = -grad

    The damping parameter mu is adapted:
    - Decreased when full Newton steps succeed (energy decreases)
    - Increased when the Hessian is indefinite or steps fail

    Convergence is declared when EITHER
        |grad|        < gtol  (absolute), OR
        |grad|/|grad0| < gtol  (relative to initial gradient).
    The relative criterion matters at low twist angles where the absolute
    gradient norm at the energy minimum can be O(1-10) due to large total
    energies, making any reasonable absolute gtol unreachable.

    The elastic Hessian is constant (precomputed). Only the GSFE Hessian
    changes each iteration. The linear system is solved with sparse LU.
    """
    from scipy.sparse import eye as speye

    U = U0.copy()
    E, grad = energy_func(U)
    gnorm = np.linalg.norm(grad)
    gnorm0 = max(gnorm, 1.0)  # avoid div-by-zero if started at minimum
    n_sol = len(U)
    nit = 0
    nfev = 1
    t_start = perf_counter()

    def converged(gn: float) -> bool:
        return gn < gtol or (gn / gnorm0) < gtol

    # Initial damping: scale relative to the Hessian diagonal
    mu = 1e-4 * (energy_func.K1 + energy_func.G1)
    mu_min = 1e-10
    mu_max = 1e10
    I_sp = speye(n_sol, format="csr")

    for nit in range(1, max_iter + 1):
        if converged(gnorm):
            break

        # Build Hessian + damping
        H = energy_func.hessian(U)
        H_damped = H + mu * I_sp

        # Solve for Newton direction
        try:
            dU = spsolve(H_damped, -grad)
        except Exception:
            mu *= 10
            continue

        # Check descent direction
        slope = grad.dot(dU)
        if slope > 0:
            # Not a descent direction — increase damping and retry
            mu *= 10
            mu = min(mu, mu_max)
            continue

        # Try full Newton step
        U_trial = U + dU
        E_new, grad_new = energy_func(U_trial)
        nfev += 1

        if E_new < E:
            # Accept step
            actual_reduction = E - E_new
            predicted_reduction = -slope - 0.5 * dU.dot(H @ dU)
            rho = actual_reduction / max(abs(predicted_reduction), 1e-30)

            U = U_trial
            E = E_new
            grad = grad_new
            gnorm = np.linalg.norm(grad)

            # Adapt damping
            if rho > 0.75:
                mu = max(mu / 3, mu_min)
            elif rho < 0.25:
                mu *= 2
        else:
            # Reject step — increase damping
            mu *= 4
            mu = min(mu, mu_max)

        if display and (nit % 5 == 0 or nit <= 3):
            elapsed = perf_counter() - t_start
            print(f"  iter {nit:4d}: E = {E:.4f}, |grad| = {gnorm:.2e}, "
                  f"mu = {mu:.2e}, t = {elapsed:.1f}s")

    success = converged(gnorm)
    message = "converged" if success else f"max iterations ({max_iter}) reached"
    return {
        "x": U, "fun": E, "jac": grad, "nit": nit, "nfev": nfev,
        "message": message, "success": success,
    }


def _pseudo_dynamics_solve(energy_func: RelaxationEnergy, U0: np.ndarray,
                           max_iter: int, gtol: float, beta: float,
                           dt0: float | None, display: bool) -> dict:
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

    Per-iteration cost: this solver rebuilds the sparse Hessian and runs
    a sparse direct LU factorization at every step, the same as Newton.
    For large meshes / many layers (n_sol >> 10^4) this becomes much
    slower per iteration than L-BFGS-B (which only does Hessian-vector
    products). Use 'pseudo_dynamics' specifically when you need the
    extra robustness on stiff multi-layer / low-twist cases where Newton
    stalls; for routine bilayer relaxation Newton is usually faster.

    This solver ports the algorithm of
    docs_internal/relaxation_code_2D_solver_periodic_ver6.m, simplified
    for the periodic-mesh case (no normal/tangent edge BC machinery,
    Amat0 = identity).
    """
    from scipy.sparse import eye as speye

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
        return gn < gtol or (gn / gnorm0) < gtol

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
        if converged(gnorm):
            break

        # Build the implicit Hessian at the current iterate. The Hessian
        # has elastic (constant) + GSFE (vertex-diagonal at U) parts.
        H = energy_func.hessian(U)
        A = M + (beta * dt) * H
        rhs = -dt * grad

        # Sparse direct solve. The system is well-posed because M is
        # positive definite even when H is indefinite or near-singular.
        try:
            dU = spsolve(A, rhs)
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
            print(f"  iter {nit:4d}: E = {E:.4f}, |grad| = {gnorm:.2e}, "
                  f"dt = {dt:.2e}, t = {elapsed:.1f}s")

        if count_conv >= count_conv_required:
            break

    success = converged(gnorm) and count_conv >= count_conv_required
    if success:
        message = "converged"
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


class RelaxationSolver:
    """Solver for atomic relaxation in twisted 2D heterostructures."""

    def __init__(self, config: SolverConfig | None = None):
        self.config = config or SolverConfig()

    def solve(
        self,
        material1: Material,
        material2: Material,
        theta_twist: float,
        nlayer1: int = 1,
        nlayer2: int = 1,
        delta: float | None = None,
        theta0: float = 0.0,
        initial_solution: np.ndarray | None = None,
        constraints: "PinnedConstraints | None" = None,
    ) -> RelaxationResult:
        """Solve the relaxation problem.

        Parameters
        ----------
        material1, material2 : Material
            Materials for the two stacks.
        theta_twist : float
            Twist angle in degrees.
        nlayer1, nlayer2 : int
            Number of layers in each stack.
        delta : float or None
            Lattice mismatch. If None, computed from material lattice constants.
        theta0 : float
            Lattice orientation angle in degrees.
        initial_solution : ndarray or None
            Initial guess. If None, starts from zero.
        constraints : PinnedConstraints or None
            If set, pins certain DOFs to fixed displacements while
            optimizing the rest. Build via PinningMap.build_constraints().
        """
        cfg = self.config

        if delta is None:
            delta = material1.lattice_constant / material2.lattice_constant - 1.0

        lattice = HexagonalLattice(alpha=material2.lattice_constant, theta0=theta0)
        geometry = MoireGeometry(lattice, theta_twist=theta_twist, delta=delta)

        if cfg.display:
            print(f"Moire wavelength: {geometry.wavelength:.2f} nm")
            print(f"Twist angle: {theta_twist:.4f} deg, delta: {delta:.6f}")

        mesh = MoireMesh.generate(
            geometry,
            pixel_size=cfg.pixel_size,
            n_scale=cfg.n_scale,
            min_points=cfg.min_mesh_points,
        )
        if cfg.display:
            print(f"Mesh: {mesh.n_vertices} vertices, {mesh.n_triangles} triangles")

        disc = PeriodicDiscretization(mesh, geometry)
        conv = disc.build_conversion_matrices(nlayer1=nlayer1, nlayer2=nlayer2)

        gsfe_interface = GSFESurface(material1.gsfe_coeffs)
        gsfe_flake1 = GSFESurface(material1.gsfe_coeffs) if nlayer1 > 1 else None
        gsfe_flake2 = GSFESurface(material2.gsfe_coeffs) if nlayer2 > 1 else None

        I1_vect = J1_vect = I2_vect = J2_vect = None
        if nlayer1 > 1 and material1.stacking_func is not None:
            I1_vect = np.array([material1.stacking_func(k)[0] for k in range(1, nlayer1)])
            J1_vect = np.array([material1.stacking_func(k)[1] for k in range(1, nlayer1)])
        if nlayer2 > 1 and material2.stacking_func is not None:
            I2_vect = np.array([material2.stacking_func(k)[0] for k in range(1, nlayer2)])
            J2_vect = np.array([material2.stacking_func(k)[1] for k in range(1, nlayer2)])

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
                res = _newton_solve(energy_func, U0, cfg.max_iter, cfg.gtol, cfg.display)
            else:
                res = _pseudo_dynamics_solve(
                    energy_func, U0, cfg.max_iter, cfg.gtol,
                    beta=cfg.beta, dt0=cfg.dt0, display=cfg.display,
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
            # Note: `disp`/`iprint` options are deprecated in SciPy 1.18 for
            # L-BFGS-B, so we do not pass them — the solver is silent by default.
            options = {"maxiter": cfg.max_iter, "gtol": cfg.gtol}
            if cfg.method == "L-BFGS-B":
                options["maxcor"] = 20
                options["maxls"] = 40
                options["ftol"] = 1e-15
            result = minimize(
                energy_func, U0, method=cfg.method, jac=True, options=options,
            )

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
            material1=material1, material2=material2,
            displacement_x1=ux1, displacement_y1=uy1,
            displacement_x2=ux2, displacement_y2=uy2,
            total_energy=result.fun, unrelaxed_energy=E0,
            gsfe_map=emaps["gsfe_interface"],
            elastic_map1=emaps["elastic_1"],
            elastic_map2=emaps["elastic_2"],
            solution_vector=U_opt, optimizer_result=result,
        )
