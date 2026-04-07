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
        - 'L-BFGS-B': Gradient-only quasi-Newton. Slower convergence but no Hessian needed.
    max_iter : int
        Maximum number of optimizer iterations.
    gtol : float
        Gradient norm tolerance for convergence.
    pixel_size : float
        Target mesh element size in nm.
    n_scale : int
        Number of moire unit cells in each direction.
    display : bool
        If True, print progress during optimization.
    """

    method: str = "newton"
    max_iter: int = 200
    gtol: float = 1e-6
    pixel_size: float = 0.2
    n_scale: int = 1
    display: bool = True
    min_mesh_points: int = 100


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
        fix_top: bool = False,
        fix_bottom: bool = False,
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
            Mutually exclusive with fix_top / fix_bottom.
        fix_top : bool
            Pin all DOFs of the topmost layer (stack 1, layer 0) to zero.
            Use this to clamp the upper free surface of the heterostructure
            and approximate a semi-infinite top.
        fix_bottom : bool
            Pin all DOFs of the bottommost layer (stack 2, layer nlayer2-1)
            to zero. Use this to clamp the substrate's free surface and
            approximate a semi-infinite bottom — typical for simulating a
            twisted flake on a thick substrate (e.g. graphene on graphite).
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

        # Outer-layer clamps build a PinnedConstraints automatically.
        if (fix_top or fix_bottom):
            if constraints is not None:
                raise ValueError(
                    "fix_top / fix_bottom cannot be combined with an explicit "
                    "constraints argument; build a single PinnedConstraints "
                    "manually if you need both kinds of pinning."
                )
            from .discretization import build_outer_layer_constraints
            constraints = build_outer_layer_constraints(
                conv, fix_top=fix_top, fix_bottom=fix_bottom,
            )

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
        if cfg.method == "newton":
            res = _newton_solve(energy_func, U0, cfg.max_iter, cfg.gtol, cfg.display)

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
