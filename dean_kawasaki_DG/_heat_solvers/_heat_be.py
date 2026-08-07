from firedrake import *

from dean_kawasaki_DG._forms import cg_diffusion_form, sipdg_diffusion_form
from dean_kawasaki_DG._solver_base import SolverBase

import numpy as np


def _as_tuple(args):
    if isinstance(args, tuple):
        return args
    if isinstance(args, list):
        return tuple(args)
    return (args,)


class _HeatBE(SolverBase):
    family = None

    def __init__(
        self,
        mesh,
        degree: int,
        dt: float,
        args: tuple,
        variant: str = "equispaced",
        solver_parameters=None,
    ):
        super().__init__(mesh, degree, dt, args, variant)
        parsed = _as_tuple(args)
        self._kappa = float(parsed[0])
        self._eta = float(parsed[1]) if self.family == "DG" else None

        self._V = FunctionSpace(self._mesh, self.family, self._degree, self._variant)
        u = TrialFunction(self._V)
        v = TestFunction(self._V)
        dtc = Constant(self._dt)

        self._u0 = Function(self._V, name="heat")
        self._u_tmp = Function(self._V, name="heat_tmp")

        if self.family == "CG":
            diffusion = cg_diffusion_form(u, v)
        else:
            diffusion = sipdg_diffusion_form(self._mesh, u, v, self._eta)

        a = inner(u, v) * dx + dtc * self._kappa * diffusion
        L = inner(self._u0, v) * dx

        params = solver_parameters or {
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu",
        }
        problem = LinearVariationalProblem(a, L, self._u_tmp, constant_jacobian=True)
        self._solver = LinearVariationalSolver(problem, solver_parameters=params)
        self.solution = self._u0

        coords = SpatialCoordinate(self._mesh)
        self.coordinates = coords
        if len(coords) >= 1:
            self.x = coords[0]
        if len(coords) >= 2:
            self.y = coords[1]

    def interpolate_initial_condition(self):
        if self.initial_condition is None:
            raise ValueError("You have not provided an initial condition")
        self._u0.interpolate(self.initial_condition)

    def _solve(self, t_final: float, save_steps: None | set[int]):
        self.interpolate_initial_condition()
        saved = []
        if save_steps is not None and 0 in save_steps:
            saved.append(self._u0.copy(deepcopy=True))
        for step in range(1, self.get_numsteps(t_final) + 1):
            self._solver.solve()
            self._u0.assign(self._u_tmp)
            if save_steps is not None and step in save_steps:
                saved.append(self._u0.copy(deepcopy=True))
        if save_steps is not None:
            return saved
        return self._u0


class HEAT_CG_BE(_HeatBE):
    """Backward Euler heat solver with continuous Galerkin elements."""

    family = "CG"


class HEAT_SIPDG_BE(_HeatBE):
    """Backward Euler heat solver with symmetric interior penalty dG elements."""

    family = "DG"


HEAT_CG_BE_1D = HEAT_CG_BE
HEAT_CG_BE_2D = HEAT_CG_BE
HEAT_SIPDG_BE_1D = HEAT_SIPDG_BE
HEAT_SIPDG_BE_2D = HEAT_SIPDG_BE
