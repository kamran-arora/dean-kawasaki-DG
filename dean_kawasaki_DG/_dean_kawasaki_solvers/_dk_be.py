import numpy as np
from firedrake import (
    Constant,
    Function,
    FunctionSpace,
    LinearVariationalProblem,
    LinearVariationalSolver,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    dx,
    inner,
)

from dean_kawasaki_DG._forms import cg_diffusion_form, sipdg_diffusion_form
from dean_kawasaki_DG._noise import DeanKawasakiNoise
from dean_kawasaki_DG._solver_base import SolverBase

from firedrake.petsc import PETSc


def _as_tuple(args):
    if isinstance(args, tuple):
        return args
    if isinstance(args, list):
        return tuple(args)
    return (args,)


class _DKBE(SolverBase):
    family = None

    def __init__(
        self,
        mesh,
        degree: int,
        dt: float,
        args: tuple,
        variant: str = "equispaced",
        SEED: int | None = None,
        *,
        noise_gradient: str = "broken",
        noise_sampling_backend: str = "cell",
        solver_parameters=None,
    ):
        super().__init__(mesh, degree, dt, args, variant)
        parsed = _as_tuple(args)
        self._kappa = float(parsed[0])
        self._num_particles = int(parsed[1])
        self._eta = float(parsed[2]) if self.family == "DG" else None
        self._seed = SEED
        self.noise_gradient = noise_gradient
        self.noise_sampling_backend = noise_sampling_backend

        self._V_rho = FunctionSpace(
            self._mesh, self.family, self._degree, self._variant
        )
        rho = TrialFunction(self._V_rho)
        v = TestFunction(self._V_rho)
        dtc = Constant(self._dt)

        self._rho0 = Function(self._V_rho, name="rho")
        self._rho_tmp = Function(self._V_rho, name="rho_tmp")
        self._rho_pos = Function(self._V_rho, name="rho_positive")

        include_jump_terms = self.family == "DG"
        self.noise = DeanKawasakiNoise(
            self._mesh,
            self._V_rho,
            self._degree,
            self._dt,
            self._kappa,
            self._num_particles,
            self._variant,
            gradient=noise_gradient,
            include_jump_terms=include_jump_terms,
            seed=SEED,
            sampling_backend=noise_sampling_backend,
        )

        if self.family == "CG":
            diffusion = cg_diffusion_form(rho, v)
        else:
            diffusion = sipdg_diffusion_form(self._mesh, rho, v, self._eta)

        a = inner(rho, v) * dx + dtc * self._kappa * diffusion
        L = inner(self._rho0, v) * dx + self.noise.variational_form(v)

        params = solver_parameters or {
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu",
        }
        problem = LinearVariationalProblem(a, L, self._rho_tmp, constant_jacobian=True)
        self._solver = LinearVariationalSolver(problem, solver_parameters=params)
        self.solution = self._rho0

        coords = SpatialCoordinate(self._mesh)
        self.coordinates = coords
        if len(coords) >= 1:
            self.x = coords[0]
        if len(coords) >= 2:
            self.y = coords[1]

    def interpolate_initial_condition(self):
        if self.initial_condition is None:
            raise ValueError("You have not provided an initial condition")
        self._rho0.interpolate(self.initial_condition)

    def _update_positive_part(self):
        self._rho_pos.dat.data[:] = np.maximum(self._rho0.dat.data, 0.0)

    def _solve(self, t_final: float, save_steps: None | set[int]):
        with PETSc.Log.Event("interpolate ic"):
            self.interpolate_initial_condition()
        saved = []
        if save_steps is not None and 0 in save_steps:
            saved.append(self._rho0.copy(deepcopy=True))
        for step in range(1, self.get_numsteps(t_final) + 1):
            with PETSc.Log.Event("update positive part"):
                self._update_positive_part()
            with PETSc.Log.Event("sample noise"):
                self.noise.sample(self._rho_pos)
            with PETSc.Log.Event("solve"):
                self._solver.solve()
            with PETSc.Log.Event("swap rho0 and rho_tmp"):
                self._rho0.assign(self._rho_tmp)
            if save_steps is not None and step in save_steps:
                with PETSc.Log.Event("append sol to list"):
                    saved.append(self._rho0.copy(deepcopy=True))
        if save_steps is not None:
            return saved
        return self._rho0


class DK_CG_BE(_DKBE):
    """Backward Euler Dean-Kawasaki solver with continuous Galerkin elements."""

    family = "CG"


class DK_SIPDG_BE(_DKBE):
    """Backward Euler Dean-Kawasaki solver with symmetric interior penalty dG."""

    family = "DG"


DK_CG_BE_1D = DK_CG_BE
DK_CG_BE_2D = DK_CG_BE
DK_SIPDG_BE_1D = DK_SIPDG_BE
DK_SIPDG_BE_2D = DK_SIPDG_BE
