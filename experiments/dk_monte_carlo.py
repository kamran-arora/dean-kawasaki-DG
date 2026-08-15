import argparse
import math
import multiprocessing as mp
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from timeit import default_timer as timer

import numpy as np  # noqa: E402

"""
A standalone Monte-Carlo script that can be run on a HPC cluster
"""

# Keep the mesh-resolution argument away from PETSc's option parser.
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
try:
    from firedrake import (
        COMM_WORLD,
        And,
        CellSize,
        FacetNormal,
        FiniteElement,
        Function,
        FunctionSpace,
        JacobianDeterminant,
        LinearVariationalProblem,
        LinearVariationalSolver,
        PeriodicIntervalMesh,
        SpatialCoordinate,
        TensorFunctionSpace,
        TestFunction,
        TrialFunction,
        VectorFunctionSpace,
        assemble,
        avg,
        conditional,
        cos,
        dS,
        dx,
        exp,
        grad,
        inner,
        jump,
        sin,
    )
    from firedrake.petsc import PETSc
    from pyop2 import op2
    from ufl.geometry import QuadratureWeight
finally:
    sys.argv = _original_argv
    del _original_argv


# Parameters

LENGTH = 2.0 * np.pi
DT = 0.001
FINAL_TIME = 0.1
KAPPA = 1.0
NUM_PARTICLES = 500_000
SIP_PENALTY = 10.0
POLYNOMIAL_ORDER = 1
ELEMENT_VARIANT = "equispaced"
NOISE_GRADIENT = "full"
MASTER_SEED = 123
TOTAL_REALISATIONS = 60_000
CI_Z = 1.96


def initial_density(x):
    """Initial probability density rho_0 on [0, 2*pi)."""
    return (
        1.0
        + 0.35 * sin(2.0 * x + 0.4)
        + 0.25 * cos(3.0 * x - 0.7)
    ) / (2.0 * np.pi)


@dataclass(frozen=True)
class Observable:
    """A test function and its exact particle second moment."""

    name: str
    description: str
    expression: Callable
    exact_particle_second_moment: float


def test_function(x):
    return sin(2.0 * x + 1.1) + 0.8 * cos(3.0 * x - 0.4)


def periodic_bump(x):
    return exp(1.5 * (cos(x - 0.7) - 1.0))


def phase_fourier(x):
    return sin(2.0 * x + 0.37) + 0.6 * cos(3.0 * x - 0.81)


def sin_mode_2(x):
    return sin(2.0 * x)


def interval_indicator(x, left, right):
    return conditional(And(x >= left, x <= right), 1.0, 0.0)


def indicator_half(x):
    return interval_indicator(x, np.pi / 2.0, 3.0 * np.pi / 2.0)


OBSERVABLES = (
    Observable(
        name="periodic_bump",
        description="exp(1.5 * (cos(x - 0.7) - 1))",
        expression=periodic_bump,
        exact_particle_second_moment=0.022431087773995044,
    ),
    Observable(
        name="phase_fourier",
        description="sin(2*x + 0.37) + 0.6*cos(3*x - 0.81)",
        expression=phase_fourier,
        exact_particle_second_moment=0.4255817180632473,
    ),
    Observable(
        name="original_fourier",
        description="sin(2*x + 1.1) + 0.8*cos(3*x - 0.4)",
        expression=test_function,
        exact_particle_second_moment=0.5424398737104824,
    ),
    Observable(
        name="sin_mode_2",
        description="sin(2*x)",
        expression=sin_mode_2,
        exact_particle_second_moment=0.2753355179413845,
    ),
    Observable(
        name="indicator_half",
        description="1_[pi/2, 3*pi/2]",
        expression=indicator_half,
        exact_particle_second_moment=0.07245614571389715,
    ),
)


def validate_observables(observables=None):
    if observables is None:
        observables = OBSERVABLES
    if not 1 <= len(observables) <= 5:
        raise ValueError("Configure between one and five OBSERVABLES")

    names = [observable.name for observable in observables]
    if len(set(names)) != len(names):
        raise ValueError("Observable names must be unique")
    for observable in observables:
        if not observable.name.strip():
            raise ValueError("Observable names cannot be empty")
        if not callable(observable.expression):
            raise ValueError(f"Observable {observable.name!r} is not callable")
        exact = observable.exact_particle_second_moment
        if not np.isfinite(exact) or exact < 0.0:
            raise ValueError(
                f"Observable {observable.name!r} has an invalid exact moment"
            )
    return tuple(observables)


@dataclass
class RunningStatistics:
    """Numerically stable sufficient statistics for scalar samples."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, value):
        value = float(value)
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def merge(self, other):
        if other.count == 0:
            return self
        if self.count == 0:
            self.count = other.count
            self.mean = other.mean
            self.m2 = other.m2
            self.minimum = other.minimum
            self.maximum = other.maximum
            return self

        combined_count = self.count + other.count
        delta = other.mean - self.mean
        self.mean += delta * other.count / combined_count
        self.m2 += (
            other.m2
            + delta * delta * self.count * other.count / combined_count
        )
        self.count = combined_count
        self.minimum = min(self.minimum, other.minimum)
        self.maximum = max(self.maximum, other.maximum)
        return self

    @property
    def sample_variance(self):
        if self.count < 2:
            return math.nan
        return self.m2 / (self.count - 1)

    @property
    def sample_standard_deviation(self):
        return math.sqrt(self.sample_variance)

    @property
    def standard_error(self):
        return self.sample_standard_deviation / math.sqrt(self.count)


@dataclass
class WorkerResult:
    worker_id: int
    statistics: dict[str, RunningStatistics]
    elapsed_seconds: float


def split_realisations(total, workers):
    """Split ``total`` runs as evenly as possible over all workers."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    if total < workers:
        raise ValueError("TOTAL_REALISATIONS must be at least mp.cpu_count()")
    quotient, remainder = divmod(total, workers)
    return [quotient + (worker < remainder) for worker in range(workers)]


def uncertainty_summary(statistics, exact_moment, z=CI_Z):
    """Return uncertainty diagnostics for the DK-minus-particle estimator."""
    if statistics.count < 2:
        raise ValueError("At least two realisations are required")

    standard_deviation = statistics.sample_standard_deviation
    standard_error = statistics.standard_error
    halfwidth = z * standard_error
    signed_difference = statistics.mean - exact_moment
    weak_error = abs(signed_difference)
    signed_low = signed_difference - halfwidth
    signed_high = signed_difference + halfwidth

    if signed_low <= 0.0 <= signed_high:
        absolute_low = 0.0
        absolute_high = max(abs(signed_low), abs(signed_high))
    else:
        absolute_low = min(abs(signed_low), abs(signed_high))
        absolute_high = max(abs(signed_low), abs(signed_high))

    relative_standard_error = (
        standard_error / weak_error if weak_error > 0.0 else math.inf
    )
    signal_to_noise = weak_error / standard_error if standard_error > 0.0 else math.inf

    return {
        "dk_moment": statistics.mean,
        "exact_particle_moment": exact_moment,
        "signed_difference": signed_difference,
        "weak_error": weak_error,
        "sample_standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "ci_halfwidth": halfwidth,
        "dk_ci_low": statistics.mean - halfwidth,
        "dk_ci_high": statistics.mean + halfwidth,
        "signed_ci_low": signed_low,
        "signed_ci_high": signed_high,
        "absolute_ci_low": absolute_low,
        "absolute_ci_high": absolute_high,
        "relative_standard_error": relative_standard_error,
        "signal_to_noise": signal_to_noise,
        "signed_ci_contains_zero": signed_low <= 0.0 <= signed_high,
    }


class CellLocalDeanKawasakiNoise:
    """Sample DG(p-1) Dean-Kawasaki noise without global matrix assembly."""

    def __init__(
        self,
        mesh,
        density_space,
        degree,
        dt,
        kappa,
        num_particles,
        variant,
        gradient,
        seed,
    ):
        if degree < 1:
            raise ValueError("POLYNOMIAL_ORDER must be at least one")
        if gradient not in {"broken", "full"}:
            raise ValueError("NOISE_GRADIENT must be 'broken' or 'full'")
        if np.issubdtype(np.dtype(PETSc.ScalarType), np.complexfloating):
            raise NotImplementedError("The cell-local sampler needs real PETSc scalars")

        self.mesh = mesh
        self.density_space = density_space
        self.degree = degree
        self.gradient = gradient
        self.variant = variant
        self.coefficient = math.sqrt(2.0 * kappa * dt / num_particles)
        self.quadrature_degree = max(2 * degree + 3, 6)
        self.rng = np.random.default_rng(seed)

        self.space = VectorFunctionSpace(mesh, "DG", degree - 1, variant=variant)
        self.field = Function(self.space, name="dk_noise")
        self._rho_weight = Function(density_space)
        self._unit_noise = Function(self.space)

        if degree == 1:
            self._setup_diagonal_sampler()
        else:
            self._setup_factor_sampler()

    def _quadrature_data(self):
        quadrature_element = FiniteElement(
            "Quadrature",
            self.mesh.ufl_cell(),
            degree=self.quadrature_degree,
            quad_scheme="default",
        )
        quadrature_space = FunctionSpace(self.mesh, quadrature_element)
        physical_weights = Function(
            quadrature_space, name="physical_quadrature_weights"
        )
        physical_weights.interpolate(
            QuadratureWeight(self.mesh) * abs(JacobianDeterminant(self.mesh))
        )
        quadrature_points = np.asarray(
            quadrature_space.finat_element.fiat_equivalent._points
        )
        return quadrature_space, physical_weights, quadrature_points

    @staticmethod
    def _tabulate_basis(space, quadrature_points, dimension):
        zero_derivative = (0,) * dimension
        return np.asarray(
            space.finat_element.fiat_equivalent.tabulate(
                0, quadrature_points
            )[zero_derivative],
            dtype=PETSc.ScalarType,
        )

    def _global_array(self, values, name):
        values = np.ascontiguousarray(values).reshape(-1)
        return op2.Global(
            values.size,
            values,
            PETSc.ScalarType,
            name,
            comm=self.mesh.comm,
        )

    def _setup_diagonal_sampler(self):
        _, self._physical_weights, points = self._quadrature_data()
        density_basis = self._tabulate_basis(
            self.density_space, points, self.mesh.topological_dimension()
        )
        num_density_dofs, num_points = density_basis.shape
        dimension = self.mesh.geometric_dimension()
        self._density_basis = self._global_array(density_basis, "dk_density_basis")

        kernel = op2.Kernel(
            self._diagonal_kernel_code(num_density_dofs, num_points, dimension),
            "dk_p1_noise_diagonal",
            requires_zeroed_output_arguments=False,
        )
        self._sample_loop = op2.ParLoop(
            kernel,
            self.mesh.cell_set,
            self.field.dat(op2.WRITE, self.field.cell_node_map()),
            self._rho_weight.dat(op2.READ, self._rho_weight.cell_node_map()),
            self._physical_weights.dat(
                op2.READ, self._physical_weights.cell_node_map()
            ),
            self._unit_noise.dat(op2.READ, self._unit_noise.cell_node_map()),
            self._density_basis(op2.READ),
        )

    def _setup_factor_sampler(self):
        scalar_noise_space = FunctionSpace(
            self.mesh, "DG", self.degree - 1, variant=self.variant
        )
        _, self._physical_weights, points = self._quadrature_data()
        dimension = self.mesh.topological_dimension()
        noise_basis = self._tabulate_basis(scalar_noise_space, points, dimension)
        density_basis = self._tabulate_basis(self.density_space, points, dimension)

        num_noise_dofs, num_points = noise_basis.shape
        num_density_dofs, density_num_points = density_basis.shape
        if density_num_points != num_points:
            raise RuntimeError("Density and noise quadrature tables do not match")

        self._noise_basis = self._global_array(noise_basis, "dk_noise_basis")
        self._density_basis = self._global_array(density_basis, "dk_density_basis")

        mass_factor_space = TensorFunctionSpace(
            self.mesh,
            "DG",
            0,
            shape=(num_noise_dofs, num_noise_dofs),
        )
        self._mass_factor = Function(
            mass_factor_space, name="dk_local_mass_cholesky"
        )
        mass_kernel = op2.Kernel(
            self._mass_factor_kernel_code(num_noise_dofs, num_points),
            "dk_mass_factor",
            requires_zeroed_output_arguments=False,
        )
        mass_loop = op2.ParLoop(
            mass_kernel,
            self.mesh.cell_set,
            self._mass_factor.dat(op2.WRITE, self._mass_factor.cell_node_map()),
            self._physical_weights.dat(
                op2.READ, self._physical_weights.cell_node_map()
            ),
            self._noise_basis(op2.READ),
        )
        mass_loop()

        factor_kernel = op2.Kernel(
            self._factor_kernel_code(
                num_noise_dofs,
                num_density_dofs,
                num_points,
                self.mesh.geometric_dimension(),
            ),
            "dk_noise_factor",
            requires_zeroed_output_arguments=False,
        )
        self._sample_loop = op2.ParLoop(
            factor_kernel,
            self.mesh.cell_set,
            self.field.dat(op2.WRITE, self.field.cell_node_map()),
            self._rho_weight.dat(op2.READ, self._rho_weight.cell_node_map()),
            self._physical_weights.dat(
                op2.READ, self._physical_weights.cell_node_map()
            ),
            self._unit_noise.dat(op2.READ, self._unit_noise.cell_node_map()),
            self._mass_factor.dat(op2.READ, self._mass_factor.cell_node_map()),
            self._noise_basis(op2.READ),
            self._density_basis(op2.READ),
        )

    @staticmethod
    def _diagonal_kernel_code(num_density_dofs, num_points, dimension):
        return f"""
#include <math.h>

void dk_p1_noise_diagonal(
    PetscScalar *restrict output,
    const PetscScalar *restrict density,
    const PetscScalar *restrict weights,
    const PetscScalar *restrict normal,
    const PetscScalar *restrict density_basis)
{{
    const int nrho = {num_density_dofs};
    const int nq = {num_points};
    const int dimension = {dimension};
    PetscScalar cell_mass = 0.0;
    PetscScalar cell_volume = 0.0;

    for (int q = 0; q < nq; ++q) {{
        PetscScalar density_value = 0.0;
        for (int a = 0; a < nrho; ++a)
            density_value += density[a] * density_basis[a*nq + q];
        cell_mass += weights[q] * density_value;
        cell_volume += weights[q];
    }}

    const PetscScalar average_density = cell_mass / cell_volume;
    const PetscScalar scale = sqrt(fmax(average_density, 0.0) / cell_volume);
    for (int component = 0; component < dimension; ++component)
        output[component] = scale * normal[component];
}}
"""

    @staticmethod
    def _mass_factor_kernel_code(num_noise_dofs, num_points):
        return f"""
#include <math.h>

void dk_mass_factor(
    PetscScalar *restrict factor,
    const PetscScalar *restrict weights,
    const PetscScalar *restrict basis)
{{
    const int n = {num_noise_dofs};
    const int nq = {num_points};
    PetscScalar mass[{num_noise_dofs * num_noise_dofs}];

    for (int i = 0; i < n*n; ++i) {{
        mass[i] = 0.0;
        factor[i] = 0.0;
    }}
    for (int q = 0; q < nq; ++q)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                mass[i*n + j] += weights[q]
                    * basis[i*nq + q] * basis[j*nq + q];

    for (int k = 0; k < n; ++k) {{
        PetscScalar diagonal = mass[k*n + k];
        for (int j = 0; j < k; ++j)
            diagonal -= factor[k*n + j] * factor[k*n + j];
        factor[k*n + k] = sqrt(diagonal);

        for (int i = k + 1; i < n; ++i) {{
            PetscScalar entry = mass[i*n + k];
            for (int j = 0; j < k; ++j)
                entry -= factor[i*n + j] * factor[k*n + j];
            factor[i*n + k] = entry / factor[k*n + k];
        }}
    }}
}}
"""

    @staticmethod
    def _factor_kernel_code(
        num_noise_dofs,
        num_density_dofs,
        num_points,
        geometric_dimension,
    ):
        tolerance_factor = 64.0 * np.finfo(PETSc.RealType).eps
        return f"""
#include <math.h>

void dk_noise_factor(
    PetscScalar *restrict output,
    const PetscScalar *restrict density,
    const PetscScalar *restrict weights,
    const PetscScalar *restrict normal,
    const PetscScalar *restrict mass_factor,
    const PetscScalar *restrict noise_basis,
    const PetscScalar *restrict density_basis)
{{
    const int n = {num_noise_dofs};
    const int nrho = {num_density_dofs};
    const int nq = {num_points};
    const int dimension = {geometric_dimension};
    PetscScalar matrix[{num_noise_dofs * num_noise_dofs}];
    PetscScalar factor[{num_noise_dofs * num_noise_dofs}];
    PetscScalar permuted[{num_noise_dofs * geometric_dimension}];
    PetscScalar rhs[{num_noise_dofs * geometric_dimension}];
    PetscScalar work[{num_noise_dofs}];
    int permutation[{num_noise_dofs}];

    for (int i = 0; i < n*n; ++i) {{
        matrix[i] = 0.0;
        factor[i] = 0.0;
    }}
    for (int i = 0; i < n; ++i)
        permutation[i] = i;

    for (int q = 0; q < nq; ++q) {{
        PetscScalar positive_density = 0.0;
        for (int a = 0; a < nrho; ++a)
            positive_density += density[a] * density_basis[a*nq + q];
        positive_density = fmax(positive_density, 0.0);

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                matrix[i*n + j] += weights[q] * positive_density
                    * noise_basis[i*nq + q] * noise_basis[j*nq + q];
    }}

    PetscScalar maximum_diagonal = 0.0;
    for (int i = 0; i < n; ++i)
        maximum_diagonal = fmax(maximum_diagonal, matrix[i*n + i]);
    const PetscScalar tolerance =
        {tolerance_factor:.17e} * n * maximum_diagonal;

    for (int k = 0; k < n; ++k) {{
        int pivot = k;
        for (int i = k + 1; i < n; ++i)
            if (matrix[i*n + i] > matrix[pivot*n + pivot])
                pivot = i;

        if (matrix[pivot*n + pivot] <= tolerance)
            break;

        if (pivot != k) {{
            for (int j = 0; j < n; ++j) {{
                PetscScalar temporary = matrix[k*n + j];
                matrix[k*n + j] = matrix[pivot*n + j];
                matrix[pivot*n + j] = temporary;
            }}
            for (int i = 0; i < n; ++i) {{
                PetscScalar temporary = matrix[i*n + k];
                matrix[i*n + k] = matrix[i*n + pivot];
                matrix[i*n + pivot] = temporary;
            }}
            for (int j = 0; j < k; ++j) {{
                PetscScalar temporary = factor[k*n + j];
                factor[k*n + j] = factor[pivot*n + j];
                factor[pivot*n + j] = temporary;
            }}
            int temporary = permutation[k];
            permutation[k] = permutation[pivot];
            permutation[pivot] = temporary;
        }}

        factor[k*n + k] = sqrt(fmax(matrix[k*n + k], 0.0));
        for (int i = k + 1; i < n; ++i)
            factor[i*n + k] = matrix[i*n + k] / factor[k*n + k];

        for (int i = k + 1; i < n; ++i)
            for (int j = k + 1; j < n; ++j)
                matrix[i*n + j] -=
                    factor[i*n + k] * factor[j*n + k];
    }}

    for (int i = 0; i < n*dimension; ++i) {{
        permuted[i] = 0.0;
        rhs[i] = 0.0;
    }}

    for (int component = 0; component < dimension; ++component) {{
        for (int i = 0; i < n; ++i)
            for (int j = 0; j <= i; ++j)
                permuted[i*dimension + component] +=
                    factor[i*n + j] * normal[j*dimension + component];

        for (int i = 0; i < n; ++i)
            rhs[permutation[i]*dimension + component] =
                permuted[i*dimension + component];

        for (int i = 0; i < n; ++i) {{
            PetscScalar value = rhs[i*dimension + component];
            for (int j = 0; j < i; ++j)
                value -= mass_factor[i*n + j] * work[j];
            work[i] = value / mass_factor[i*n + i];
        }}
        for (int i = n - 1; i >= 0; --i) {{
            PetscScalar value = work[i];
            for (int j = i + 1; j < n; ++j)
                value -= mass_factor[j*n + i]
                    * output[j*dimension + component];
            output[i*dimension + component] =
                value / mass_factor[i*n + i];
        }}
    }}
}}
"""

    def sample(self, positive_density):
        self._rho_weight.assign(positive_density)
        self._unit_noise.dat.data[:] = self.rng.standard_normal(
            self._unit_noise.dat.data.shape
        )
        self._sample_loop()

    def variational_form(self, test):
        form = inner(self.field, grad(test)) * dx
        if self.gradient == "full":
            normal = FacetNormal(self.mesh)
            form -= inner(avg(self.field), jump(test, normal)) * dS
        return self.coefficient * form


def sipdg_diffusion_form(mesh, trial, test, penalty):
    normal = FacetNormal(mesh)
    cell_size = CellSize(mesh)
    facet_size = (cell_size("+") + cell_size("-")) / 2.0
    return (
        inner(grad(trial), grad(test)) * dx
        - inner(avg(grad(trial)), jump(test, normal)) * dS
        - inner(avg(grad(test)), jump(trial, normal)) * dS
        + penalty / facet_size * inner(jump(trial, normal), jump(test, normal)) * dS
    )


def number_of_steps(final_time, dt):
    steps = int(round(final_time / dt))
    if not np.isclose(steps * dt, final_time, rtol=0.0, atol=1.0e-12):
        raise ValueError("FINAL_TIME must be an integer multiple of DT")
    return steps


class HeatSolver:
    def __init__(self, mesh, space, dt, kappa, penalty):
        trial = TrialFunction(space)
        test = TestFunction(space)
        self._dt = dt
        self._rho = Function(space, name="heat_density")
        self._next_rho = Function(space, name="heat_density_next")
        bilinear = (
            inner(trial, test) * dx
            + dt * kappa * sipdg_diffusion_form(mesh, trial, test, penalty)
        )
        linear = inner(self._rho, test) * dx
        problem = LinearVariationalProblem(
            bilinear, linear, self._next_rho, constant_jacobian=True
        )
        self._solver = LinearVariationalSolver(
            problem,
            solver_parameters={
                "mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )

    def solve(self, initial_condition, final_time):
        self._rho.interpolate(initial_condition)
        for _ in range(number_of_steps(final_time, self._dt)):
            self._solver.solve()
            self._rho.assign(self._next_rho)
        return self._rho


class DeanKawasakiSolver:
    def __init__(
        self,
        mesh,
        space,
        degree,
        dt,
        kappa,
        num_particles,
        penalty,
        variant,
        noise_gradient,
        seed,
    ):
        trial = TrialFunction(space)
        test = TestFunction(space)
        self._dt = dt
        self._rho = Function(space, name="dk_density")
        self._next_rho = Function(space, name="dk_density_next")
        self._positive_rho = Function(space, name="dk_positive_density")
        self.noise = CellLocalDeanKawasakiNoise(
            mesh,
            space,
            degree,
            dt,
            kappa,
            num_particles,
            variant,
            noise_gradient,
            seed,
        )

        bilinear = (
            inner(trial, test) * dx
            + dt * kappa * sipdg_diffusion_form(mesh, trial, test, penalty)
        )
        linear = inner(self._rho, test) * dx + self.noise.variational_form(test)
        problem = LinearVariationalProblem(
            bilinear, linear, self._next_rho, constant_jacobian=True
        )
        self._solver = LinearVariationalSolver(
            problem,
            solver_parameters={
                "mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )

    def solve(self, initial_condition, final_time):
        self._rho.interpolate(initial_condition)
        for _ in range(number_of_steps(final_time, self._dt)):
            self._positive_rho.dat.data[:] = np.maximum(
                self._rho.dat.data_ro, 0.0
            )
            self.noise.sample(self._positive_rho)
            self._solver.solve()
            self._rho.assign(self._next_rho)
        return self._rho


def observable_weights(space, phi):
    """Assemble the dual vector for <function, interpolated phi>."""
    test = TestFunction(space)
    phi_h = Function(space, name="test_function").interpolate(phi)
    return assemble(phi_h * test * dx).dat.data_ro.copy()


def run_worker(task):
    """Run one worker and return only sufficient statistics, not paths."""
    worker_id, realisations, nx, seed = task
    started = timer()

    mesh = PeriodicIntervalMesh(ncells=nx, length=LENGTH)
    space = FunctionSpace(
        mesh, "DG", POLYNOMIAL_ORDER, variant=ELEMENT_VARIANT
    )
    (x,) = SpatialCoordinate(mesh)
    rho_initial = initial_density(x)

    heat = HeatSolver(mesh, space, DT, KAPPA, SIP_PENALTY)
    rho_mfl = heat.solve(rho_initial, FINAL_TIME)
    mfl_coefficients = rho_mfl.dat.data_ro.copy()
    observables = validate_observables()
    weights = {
        observable.name: observable_weights(
            space, observable.expression(x)
        )
        for observable in observables
    }

    dk = DeanKawasakiSolver(
        mesh,
        space,
        POLYNOMIAL_ORDER,
        DT,
        KAPPA,
        NUM_PARTICLES,
        SIP_PENALTY,
        ELEMENT_VARIANT,
        NOISE_GRADIENT,
        seed,
    )

    statistics = {
        observable.name: RunningStatistics() for observable in observables
    }
    root_n = math.sqrt(NUM_PARTICLES)
    for _ in range(realisations):
        rho_dk = dk.solve(rho_initial, FINAL_TIME)
        fluctuation = mfl_coefficients - rho_dk.dat.data_ro
        for observable in observables:
            pairing = np.dot(fluctuation, weights[observable.name])
            statistics[observable.name].update((root_n * pairing) ** 2)

    return WorkerResult(worker_id, statistics, timer() - started)


def print_results(nx, workers, results, elapsed_seconds):
    observables = validate_observables()
    combined = {
        observable.name: RunningStatistics() for observable in observables
    }
    for result in results:
        if set(result.statistics) != set(combined):
            raise ValueError("Worker returned statistics for the wrong observables")
        for observable in observables:
            combined[observable.name].merge(
                result.statistics[observable.name]
            )

    first_name = observables[0].name
    worker_counts = [
        result.statistics[first_name].count for result in results
    ]
    worker_times = [result.elapsed_seconds for result in results]
    total_count = combined[first_name].count
    if any(statistics.count != total_count for statistics in combined.values()):
        raise RuntimeError("Observable sample counts do not match")

    print("\nDean-Kawasaki Monte-Carlo results")
    print("=" * 40)
    print(f"nx                              : {nx}")
    print(f"h                               : {LENGTH / nx:.12g}")
    print(f"polynomial order                : {POLYNOMIAL_ORDER}")
    print(f"noise gradient                  : {NOISE_GRADIENT}")
    print(f"observables                     : {len(observables)}")
    print(f"realisations per observable     : {total_count}")
    print(f"worker processes                : {workers}")
    print(f"runs per worker (min, max)      : {min(worker_counts)}, {max(worker_counts)}")
    print(f"total wall time [s]             : {elapsed_seconds:.6f}")
    print(f"worker time range [s]           : {min(worker_times):.6f}, {max(worker_times):.6f}")
    summaries = {}
    for index, observable in enumerate(observables, start=1):
        statistics = combined[observable.name]
        summary = uncertainty_summary(
            statistics, observable.exact_particle_second_moment
        )
        summaries[observable.name] = summary
        worker_means = [
            result.statistics[observable.name].mean for result in results
        ]

        print("\n" + "-" * 72)
        print(f"Observable {index}/{len(observables)}: {observable.name}")
        print(f"phi                             : {observable.description}")
        print(
            "sample range                    : "
            f"[{statistics.minimum:.12g}, {statistics.maximum:.12g}]"
        )
        print(
            "worker estimator range          : "
            f"[{min(worker_means):.12g}, {max(worker_means):.12g}]"
        )
        print(f"DK second-moment estimate       : {summary['dk_moment']:.16g}")
        print(f"exact particle second moment    : {summary['exact_particle_moment']:.16g}")
        print(f"signed difference (DK-particle) : {summary['signed_difference']:.16g}")
        print(f"absolute weak error             : {summary['weak_error']:.16g}")
        print(
            "sample standard deviation       : "
            f"{summary['sample_standard_deviation']:.16g}"
        )
        print(f"standard error of DK estimate   : {summary['standard_error']:.16g}")
        print(f"95% CI half-width               : {summary['ci_halfwidth']:.16g}")
        print(
            "95% CI for DK moment            : "
            f"[{summary['dk_ci_low']:.16g}, {summary['dk_ci_high']:.16g}]"
        )
        print(
            "95% CI for signed difference    : "
            f"[{summary['signed_ci_low']:.16g}, {summary['signed_ci_high']:.16g}]"
        )
        print(
            "95% CI induced for |difference| : "
            f"[{summary['absolute_ci_low']:.16g}, {summary['absolute_ci_high']:.16g}]"
        )
        print(
            "relative standard error         : "
            f"{summary['relative_standard_error']:.6g}"
        )
        print(f"|difference| / standard error   : {summary['signal_to_noise']:.6g}")
        print(f"signed CI contains zero         : {summary['signed_ci_contains_zero']}")
    return summaries


def positive_mesh_resolution(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("NX must be a positive integer")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Standalone 1D Dean-Kawasaki Monte-Carlo experiment"
    )
    parser.add_argument("nx", type=positive_mesh_resolution, help="number of cells")
    arguments = parser.parse_args(argv)

    if COMM_WORLD.size != 1:
        raise RuntimeError("Run with plain Python, not mpiexec or srun --ntasks > 1")

    observables = validate_observables()
    procs = mp.cpu_count()
    run_counts = split_realisations(TOTAL_REALISATIONS, procs)
    child_sequences = np.random.SeedSequence(MASTER_SEED).spawn(procs)
    child_seeds = [
        int(sequence.generate_state(1, dtype=np.uint64)[0])
        for sequence in child_sequences
    ]
    tasks = [
        (worker_id, run_counts[worker_id], arguments.nx, child_seeds[worker_id])
        for worker_id in range(procs)
    ]

    print(
        f"Running {TOTAL_REALISATIONS} realisations at nx={arguments.nx} "
        f"on {procs} processes for {len(observables)} observables"
    )
    for observable in observables:
        print(
            f"  {observable.name}: {observable.description}; "
            f"exact={observable.exact_particle_second_moment:.16g}"
        )
    started = timer()
    with mp.Pool(processes=procs) as pool:
        results = pool.map(run_worker, tasks)
    print_results(arguments.nx, procs, results, timer() - started)


if __name__ == "__main__":
    mp.freeze_support()
    main()
