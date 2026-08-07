from firedrake import *
from abc import ABC, abstractmethod
from ufl.core.expr import Expr
import numpy as np


def prepare_saveat(
    saveat: None | np.ndarray,
    dt: float,
    t_final: float,
    *,
    atol: float = 1.0e-10,
) -> None | np.ndarray:
    """
    Given an array of times to saveat, prepare a set of steps (integers) to 
    store the solution at

    Args:
    - saveat: None | np.ndarray
        if None, save solution only at final time
        if np.ndarray, save solution at times specified in this array
    """
    if saveat is None:
        return None

    saveat = np.asarray(saveat, dtype=float)

    if saveat.ndim != 1:
        raise ValueError("saveat must be one-dimensional")
    if np.any(saveat < -atol):
        raise ValueError("saveat cannot contain negative times")
    if np.any(saveat > t_final + atol):
        raise ValueError("saveat cannot contain times larger than the final time")

    final_step_float = t_final / dt
    final_step = int(round(final_step_float))
    if not np.isclose(final_step_float, final_step, rtol=0.0, atol=atol):
        raise ValueError(f"final time={t_final} must be an integer multiple of dt={dt}")

    step_numbers = saveat / dt
    rounded_steps = np.rint(step_numbers).astype(int)

    valid = np.isclose(step_numbers, rounded_steps, rtol=0.0, atol=atol)

    if not np.all(valid):
        bad_times = saveat[~valid]
        raise ValueError(
            f"All saveat times must be integer multiples of dt={dt}."
            f"Bad times: {bad_times}"
        )

    return np.unique(rounded_steps)


class SolverBase(ABC):
    """
    Common methods for all solvers

    :mesh: underlying mesh
    :degree: polynomial degree
    :dt: timestep
    :args: tuple
    """

    def __init__(
        self, mesh, degree: int, dt: float, args: tuple, variant: str = "equispaced"
    ):
        self._mesh = mesh
        if not isinstance(degree, int):
            raise ValueError("degree must be an integer")
        self._degree = degree
        self._dt = dt
        self._args = args
        self._variant = variant

        # set initial condition to be None
        self.initial_condition = None

    def get_numsteps(self, t_final: float):
        num_steps = int(np.round(t_final / self._dt))
        return num_steps

    def set_initial_condition(self, expr):
        """
        Set initial condition

        expr: ufl expression
        """
        if not isinstance(expr, Expr):
            raise ValueError("Initial condition must be a ufl expression")
        self.initial_condition = expr

    def set_mixed_initial_condition(self, expr1, expr2):
        """
        Set initial condition for a mixed problem

        expr1: ufl expression
        expr2: ufl expression
        """
        if not isinstance(expr1, Expr) or not isinstance(expr2, Expr):
            raise ValueError("Initial conditions must be a ufl expression")
        self.initial_condition = (expr1, expr2)

    def initial_condition_from_particles(self, particles: np.ndarray):
        """
        Define initial condition from a set of particle positions

        Computed via solving a variational problem on the solver density space
        when available, and on DG(degree) otherwise.

        particles: shape (N, d)
        """
        # extract N
        N = particles.shape[0]
        U = getattr(self, "_V_rho", getattr(self, "_V", None))
        if U is None:
            U = FunctionSpace(self._mesh, "DG", self._degree, variant=self._variant)
        rho_p = TrialFunction(U)
        u_p = TestFunction(U)
        # vertex only mesh
        vom = VertexOnlyMesh(self._mesh, particles)
        V = FunctionSpace(vom, "DG", 0)
        f_p = Function(V).assign(1.0 / N)
        v_p = TestFunction(V)
        # rhs
        rhs = Cofunction(U.dual()).interpolate(assemble(f_p * v_p * dx))
        # solve
        self.initial_condition = Function(U)
        solve(inner(rho_p, u_p) * dx == rhs(u_p), self.initial_condition)

    @abstractmethod
    def interpolate_initial_condition(self):
        """
        Interpolate initial condition
        """
        raise NotImplementedError

    def solve(self, t_final: float, saveat: None | np.ndarray = None):
        """
        Public solve method shared by solvers
        """
        save_steps = _prepare_saveat(saveat, self._dt, t_final)
        return self._solve(t_final=t_final, save_steps=save_steps)

    @abstractmethod
    def _solve(self, t_final: float, save_steps: None | set[int]):
        """
        Solver-specific method to solve the (S)PDE
        """
        raise NotImplementedError
