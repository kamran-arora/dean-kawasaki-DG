from __future__ import annotations

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from firedrake import (
    CellVolume,
    FacetNormal,
    FiniteElement,
    Function,
    FunctionSpace,
    JacobianDeterminant,
    LinearVariationalProblem,
    LinearVariationalSolver,
    TensorFunctionSpace,
    TestFunction,
    TrialFunction,
    VectorFunctionSpace,
    assemble,
    avg,
    dS,
    dx,
    grad,
    inner,
    jump,
)
from pyop2 import op2
from ufl.geometry import QuadratureWeight

from firedrake.petsc import PETSc


def _matrix_to_csr(tensor):
    indptr, indices, data = tensor.M.handle.getValuesCSR()
    return sp.csr_matrix((data, indices, indptr))


class DeanKawasakiNoise:
    """Sample the finite-dimensional Dean-Kawasaki noise.

    The sampled field lives in ``DG(p-1)^d``. The variational form determines
    whether it acts through the broken gradient or through the full
    distributional dG gradient. ``sampling_backend="cell"`` uses PyOP2
    element-local kernels. For ``p = 1``, ``"diagonal"`` selects the original
    DG0 projection implementation. For ``p > 1``, ``"global"`` selects the
    original assembled SciPy implementation.
    """

    def __init__(
        self,
        mesh,
        density_space,
        degree: int,
        dt: float,
        kappa: float,
        num_particles: int,
        variant: str = "equispaced",
        *,
        gradient: str = "broken",
        include_jump_terms: bool = False,
        seed: int | None = None,
        quadrature_degree: int | None = None,
        sampling_backend: str = "cell",
    ):
        if gradient not in {"broken", "full"}:
            raise ValueError("gradient must be 'broken' or 'full'")
        if sampling_backend not in {"cell", "global", "diagonal"}:
            raise ValueError(
                "sampling_backend must be 'cell', 'global', or 'diagonal'"
            )
        if degree < 1:
            raise ValueError("Dean-Kawasaki noise requires degree >= 1")
        if num_particles <= 0:
            raise ValueError("num_particles must be positive")

        self.mesh = mesh
        self.density_space = density_space
        self.degree = degree
        self.gradient = gradient
        self.variant = variant
        self.sampling_backend = sampling_backend
        self.include_jump_terms = include_jump_terms and gradient == "full"
        self.coefficient = np.sqrt(2.0 * kappa * dt / num_particles)
        self.quadrature_degree = quadrature_degree or max(2 * degree + 3, 6)
        self.rng = self._make_rng(mesh.comm, seed)

        self.space = VectorFunctionSpace(mesh, "DG", degree - 1, variant=self.variant)
        self.field = Function(self.space, name="dk_noise")

        if degree == 1 and sampling_backend == "cell":
            self._setup_cell_diagonal_sampler()
        elif degree == 1:
            self._setup_diagonal_sampler()
        elif sampling_backend == "diagonal":
            raise ValueError("sampling_backend='diagonal' is only valid for degree 1")
        elif sampling_backend == "global":
            self._setup_global_factor_sampler()
        else:
            self._setup_cell_factor_sampler()

    @staticmethod
    def _make_rng(comm, seed):
        if comm.size == 1:
            return np.random.default_rng(seed)

        if seed is None:
            entropy = np.random.SeedSequence().entropy if comm.rank == 0 else None
            seed = comm.bcast(entropy, root=0)
        streams = np.random.SeedSequence(seed).spawn(comm.size)
        return np.random.default_rng(streams[comm.rank])

    def _setup_diagonal_sampler(self):
        self._sampler_kind = "diagonal"
        self._unit_noise = Function(self.space)
        self._rho_average_space = FunctionSpace(
            self.mesh, "DG", 0, variant=self.variant
        )
        self._rho_average = Function(self._rho_average_space)
        self._rho_projection_source = Function(self.density_space)

        self._cell_volume = Function(self._rho_average_space, name="cell_volume")
        self._cell_volume.interpolate(CellVolume(self.mesh))

        q = TrialFunction(self._rho_average_space)
        z = TestFunction(self._rho_average_space)
        a_proj = inner(q, z) * dx
        L_proj = inner(self._rho_projection_source, z) * dx
        problem = LinearVariationalProblem(
            a_proj, L_proj, self._rho_average, constant_jacobian=True
        )
        self._projection_solver = LinearVariationalSolver(
            problem,
            solver_parameters={
                "mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "jacobi",
            },
        )

    def _setup_cell_diagonal_sampler(self):
        self._sampler_kind = "cell-diagonal"
        self._rho_weight = Function(self.density_space)
        self._unit_noise = Function(self.space)

        if np.issubdtype(np.dtype(PETSc.ScalarType), np.complexfloating):
            raise NotImplementedError(
                "The cell-local Dean-Kawasaki noise sampler requires "
                "a real-valued PETSc build"
            )

        quadrature_element = FiniteElement(
            "Quadrature",
            self.mesh.ufl_cell(),
            degree=self.quadrature_degree,
            quad_scheme="default",
        )
        quadrature_space = FunctionSpace(self.mesh, quadrature_element)

        self._physical_quadrature_weights = Function(
            quadrature_space, name="physical_quadrature_weights"
        )
        self._physical_quadrature_weights.interpolate(
            QuadratureWeight(self.mesh) * abs(JacobianDeterminant(self.mesh))
        )

        quadrature_fiat = quadrature_space.finat_element.fiat_equivalent
        quadrature_points = np.asarray(quadrature_fiat._points)
        zero_derivative = (0,) * self.mesh.topological_dimension()
        density_basis = np.asarray(
            self.density_space.finat_element.fiat_equivalent.tabulate(
                0, quadrature_points
            )[zero_derivative],
            dtype=PETSc.ScalarType,
        )

        num_local_density_dofs, num_quadrature_points = density_basis.shape
        geometric_dimension = self.mesh.geometric_dimension()
        self._density_basis = op2.Global(
            density_basis.size,
            np.ascontiguousarray(density_basis).reshape(-1),
            PETSc.ScalarType,
            "dk_density_basis",
            comm=self.mesh.comm,
        )

        diagonal_kernel = op2.Kernel(
            self._diagonal_kernel_code(
                num_local_density_dofs,
                num_quadrature_points,
                geometric_dimension,
            ),
            "dk_p1_noise_diagonal",
            requires_zeroed_output_arguments=False,
        )
        self._diagonal_loop = op2.ParLoop(
            diagonal_kernel,
            self.mesh.cell_set,
            self.field.dat(op2.WRITE, self.field.cell_node_map()),
            self._rho_weight.dat(op2.READ, self._rho_weight.cell_node_map()),
            self._physical_quadrature_weights.dat(
                op2.READ, self._physical_quadrature_weights.cell_node_map()
            ),
            self._unit_noise.dat(op2.READ, self._unit_noise.cell_node_map()),
            self._density_basis(op2.READ),
        )

    def _setup_global_factor_sampler(self):
        if self.mesh.comm.size != 1:
            raise NotImplementedError(
                "The legacy global noise backend is only supported in serial"
            )

        self._sampler_kind = "global-factor"
        trial = TrialFunction(self.space)
        test = TestFunction(self.space)

        mass = assemble(inner(trial, test) * dx)
        mass_csr = _matrix_to_csr(mass)
        self._mass_solve = spla.factorized(mass_csr.tocsc())

        self._rho_weight = Function(self.density_space)
        self._weighted_mass_form = (
            self._rho_weight
            * inner(TrialFunction(self.space), TestFunction(self.space))
            * dx(degree=self.quadrature_degree)
        )

    def _setup_cell_factor_sampler(self):
        self._sampler_kind = "cell-factor"
        self._rho_weight = Function(self.density_space)
        self._unit_noise = Function(self.space)

        if np.issubdtype(np.dtype(PETSc.ScalarType), np.complexfloating):
            raise NotImplementedError(
                "The cell-local Dean-Kawasaki noise sampler requires "
                "a real-valued PETSc build"
            )

        scalar_noise_space = FunctionSpace(
            self.mesh, "DG", self.degree - 1, variant=self.variant
        )
        quadrature_element = FiniteElement(
            "Quadrature",
            self.mesh.ufl_cell(),
            degree=self.quadrature_degree,
            quad_scheme="default",
        )
        quadrature_space = FunctionSpace(self.mesh, quadrature_element)

        self._physical_quadrature_weights = Function(
            quadrature_space, name="physical_quadrature_weights"
        )
        self._physical_quadrature_weights.interpolate(
            QuadratureWeight(self.mesh) * abs(JacobianDeterminant(self.mesh))
        )

        quadrature_fiat = quadrature_space.finat_element.fiat_equivalent
        quadrature_points = np.asarray(quadrature_fiat._points)
        zero_derivative = (0,) * self.mesh.topological_dimension()

        noise_basis = np.asarray(
            scalar_noise_space.finat_element.fiat_equivalent.tabulate(
                0, quadrature_points
            )[zero_derivative],
            dtype=PETSc.ScalarType,
        )
        density_basis = np.asarray(
            self.density_space.finat_element.fiat_equivalent.tabulate(
                0, quadrature_points
            )[zero_derivative],
            dtype=PETSc.ScalarType,
        )

        self._num_local_noise_dofs, num_quadrature_points = noise_basis.shape
        num_local_density_dofs, density_num_quadrature_points = density_basis.shape
        if density_num_quadrature_points != num_quadrature_points:
            raise RuntimeError("Density and noise basis quadrature tables do not match")

        geometric_dimension = self.mesh.geometric_dimension()
        self._noise_basis = op2.Global(
            noise_basis.size,
            np.ascontiguousarray(noise_basis).reshape(-1),
            PETSc.ScalarType,
            "dk_noise_basis",
            comm=self.mesh.comm,
        )
        self._density_basis = op2.Global(
            density_basis.size,
            np.ascontiguousarray(density_basis).reshape(-1),
            PETSc.ScalarType,
            "dk_density_basis",
            comm=self.mesh.comm,
        )

        mass_factor_space = TensorFunctionSpace(
            self.mesh,
            "DG",
            0,
            shape=(self._num_local_noise_dofs, self._num_local_noise_dofs),
        )
        self._mass_factor = Function(mass_factor_space, name="dk_local_mass_cholesky")

        mass_kernel = op2.Kernel(
            self._mass_factor_kernel_code(
                self._num_local_noise_dofs, num_quadrature_points
            ),
            "dk_mass_factor",
            requires_zeroed_output_arguments=False,
        )
        mass_loop = op2.ParLoop(
            mass_kernel,
            self.mesh.cell_set,
            self._mass_factor.dat(op2.WRITE, self._mass_factor.cell_node_map()),
            self._physical_quadrature_weights.dat(
                op2.READ, self._physical_quadrature_weights.cell_node_map()
            ),
            self._noise_basis(op2.READ),
        )
        with PETSc.Log.Event("noise p>=2 setup local mass factors"):
            mass_loop()

        factor_kernel = op2.Kernel(
            self._factor_kernel_code(
                self._num_local_noise_dofs,
                num_local_density_dofs,
                num_quadrature_points,
                geometric_dimension,
            ),
            "dk_noise_factor",
            requires_zeroed_output_arguments=False,
        )
        self._factor_loop = op2.ParLoop(
            factor_kernel,
            self.mesh.cell_set,
            self.field.dat(op2.WRITE, self.field.cell_node_map()),
            self._rho_weight.dat(op2.READ, self._rho_weight.cell_node_map()),
            self._physical_quadrature_weights.dat(
                op2.READ, self._physical_quadrature_weights.cell_node_map()
            ),
            self._unit_noise.dat(op2.READ, self._unit_noise.cell_node_map()),
            self._mass_factor.dat(op2.READ, self._mass_factor.cell_node_map()),
            self._noise_basis(op2.READ),
            self._density_basis(op2.READ),
        )

    @staticmethod
    def _diagonal_kernel_code(
        num_density_dofs,
        num_quadrature_points,
        geometric_dimension,
    ):
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
    const int nq = {num_quadrature_points};
    const int dimension = {geometric_dimension};
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
    def _mass_factor_kernel_code(num_noise_dofs, num_quadrature_points):
        return f"""
#include <math.h>

void dk_mass_factor(
    PetscScalar *restrict factor,
    const PetscScalar *restrict weights,
    const PetscScalar *restrict basis)
{{
    const int n = {num_noise_dofs};
    const int nq = {num_quadrature_points};
    PetscScalar mass[{num_noise_dofs * num_noise_dofs}];

    for (int i = 0; i < n * n; ++i) {{
        mass[i] = 0.0;
        factor[i] = 0.0;
    }}

    for (int q = 0; q < nq; ++q)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                mass[i*n + j] += weights[q]
                    * basis[i*nq + q] * basis[j*nq + q];

    /* Unpivoted Cholesky is safe here: this is the unweighted DG mass. */
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
        num_quadrature_points,
        geometric_dimension,
    ):
        real_epsilon = np.finfo(PETSc.RealType).eps
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
    const int nq = {num_quadrature_points};
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

    /*
     * Assemble the scalar cell block
     * P_K = integral_K max(rho, 0) phi_i phi_j.
     * The same block applies independently to every vector component.
     */
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

    /*
     * Complete-pivoting Cholesky for a positive-semidefinite matrix.
     * On exit P_K[permutation, permutation] = factor factor^T up to
     * roundoff-sized eigenvalues.
     */
    PetscScalar maximum_diagonal = 0.0;
    for (int i = 0; i < n; ++i)
        maximum_diagonal = fmax(
            maximum_diagonal, matrix[i*n + i]
        );
    const PetscScalar tolerance =
        {64.0 * real_epsilon:.17e} * n * maximum_diagonal;

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
            factor[i*n + k] =
                matrix[i*n + k] / factor[k*n + k];

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

        /* Solve M_K output = rhs using the cached M_K Cholesky factor. */
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

    def sample(self, rho_positive):
        """Update ``field`` using the supplied positive density."""
        if self._sampler_kind == "diagonal":
            self._sample_diagonal(rho_positive)
        elif self._sampler_kind == "cell-diagonal":
            self._sample_cell_diagonal(rho_positive)
        elif self._sampler_kind == "global-factor":
            self._sample_global_factor(rho_positive)
        else:
            self._sample_cell_factor(rho_positive)

    def _sample_diagonal(self, rho_positive):
        with PETSc.Log.Event("noise p1 assign rho"):
            self._rho_projection_source.assign(rho_positive)
        with PETSc.Log.Event("noise p1 project rho to DG0"):
            self._projection_solver.solve()
        with PETSc.Log.Event("noise p1 random draw"):
            self._unit_noise.dat.data[:] = self.rng.standard_normal(
                self._unit_noise.dat.data.shape
            )
        with PETSc.Log.Event("noise p1 direct dat write"):
            rho_data = self._rho_average.dat.data_ro
            volume_data = self._cell_volume.dat.data_ro

            scale = np.sqrt(np.maximum(rho_data, 0.0) / volume_data)
            while scale.ndim < self._unit_noise.dat.data.ndim:
                scale = scale[..., None]

            # VectorFunctionSpace(DG0) commonly stores data as (cells, dim).
            # If Firedrake stores a scalar-looking shape, broadcasting below still works.
            self.field.dat.data[:] = scale * self._unit_noise.dat.data[:]
        # scale = sqrt(self._rho_average / CellVolume(self.mesh))
        # self.field.interpolate(scale * self._unit_noise)

    def _sample_cell_diagonal(self, rho_positive):
        with PETSc.Log.Event("noise p1 cell assign rho"):
            self._rho_weight.assign(rho_positive)
        with PETSc.Log.Event("noise p1 cell random draw"):
            self._unit_noise.dat.data[:] = self.rng.standard_normal(
                self._unit_noise.dat.data.shape
            )
        with PETSc.Log.Event("noise p1 cell-local average and scale"):
            self._diagonal_loop()

    def _sample_global_factor(self, rho_positive):
        with PETSc.Log.Event("noise p>=2 global assign rho"):
            self._rho_weight.assign(rho_positive)
        with PETSc.Log.Event("noise p>=2 global assemble weighted mass"):
            weighted_mass = assemble(self._weighted_mass_form)
        with PETSc.Log.Event("noise p>=2 global csr conversion"):
            weighted_csr = _matrix_to_csr(weighted_mass)
        with PETSc.Log.Event("noise p>=2 global dense conversion"):
            weighted = weighted_csr.toarray()
        with PETSc.Log.Event("noise p>=2 global sqrt psd"):
            factor = self._sqrt_psd(weighted)
        with PETSc.Log.Event("noise p>=2 global random draw"):
            white = self.rng.standard_normal(weighted.shape[0])
        with PETSc.Log.Event("noise p>=2 global mass solve"):
            coefficients = self._mass_solve(factor @ white)
        with PETSc.Log.Event("noise p>=2 global direct dat write"):
            self.field.dat.data.reshape(-1)[:] = coefficients

    @staticmethod
    def _sqrt_psd(matrix):
        try:
            return scipy.linalg.cholesky(matrix, lower=True, check_finite=False)
        except scipy.linalg.LinAlgError:
            eigenvalues, eigenvectors = scipy.linalg.eigh(matrix, check_finite=False)
            eigenvalues = np.clip(eigenvalues, 0.0, None)
            return eigenvectors * np.sqrt(eigenvalues)[None, :]

    def _sample_cell_factor(self, rho_positive):
        with PETSc.Log.Event("noise p>=2 cell assign rho"):
            self._rho_weight.assign(rho_positive)
        with PETSc.Log.Event("noise p>=2 cell random draw"):
            self._unit_noise.dat.data[:] = self.rng.standard_normal(
                self._unit_noise.dat.data.shape
            )
        with PETSc.Log.Event("noise p>=2 cell-local factor and mass solve"):
            self._factor_loop()

    def variational_form(self, test_function):
        form = inner(self.field, grad(test_function)) * dx
        if self.include_jump_terms:
            n = FacetNormal(self.mesh)
            form -= inner(avg(self.field), jump(test_function, n)) * dS
        return self.coefficient * form
