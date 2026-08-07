from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

ParticleMethod = Literal[
    "random-inverse-1d",
    "deterministic-quantile-1d",
    "random-grid",
    "random-grid-1d",
    "random-grid-2d",
    "deterministic-rosenblatt-2d",
    "rejection",
    "rejection-uniform",
]

Array = np.ndarray
Density1D = Callable[[Array], Array]
DensityND = Callable[..., Array]
Domain = Sequence[tuple[float, float]]

__all__ = [
    "ParticleConfiguration",
    "ParticleMethod",
    "as_generator",
    "child_seed",
    "deterministic_quantile_1d",
    "deterministic_rosenblatt_2d",
    "estimate_density_max",
    "generate_initial_particles",
    "normalise_1d",
    "random_grid",
    "random_inverse_1d",
    "rejection_uniform",
]

@dataclass(frozen=True)
class ParticleConfiguration:
    """Particle configuration for reproducible initial data."""

    particles: Array
    method: str
    seed: int | None
    requested_count: int
    actual_count: int
    dimension: int
    domain: tuple[tuple[float, float], ...]
    metadata: dict[str, object] = field(default_factory=dict)

def as_generator(seed_or_rng: int | np.random.Generator | None) -> np.random.Generator:
    """Return a NumPy Generator from an integer seed, Generator, or ``None``."""
    if isinstance(seed_or_rng, np.random.Generator):
        return seed_or_rng
    return np.random.default_rng(seed_or_rng)

def child_seed(seed: int, stream: int) -> int:
    """Derive a stable child seed for an independent random stream."""
    if stream < 0:
        raise ValueError("stream must be non-negative")
    sequence = np.random.SeedSequence([int(seed), int(stream)])
    return int(sequence.generate_state(1)[0])

def normalise_1d(
    density: Density1D,
    *,
    domain: tuple[float, float],
    grid_size: int = 100_000,
) -> tuple[Density1D, float]:
    """Return a normalised 1D density and its estimated normalising constant."""
    a, b = _validate_interval(domain)
    if grid_size <= 1:
        raise ValueError("grid_size must be greater than one")
    edges = np.linspace(a, b, grid_size + 1)
    grid = 0.5 * (edges[:-1] + edges[1:])
    values = _positive_finite(np.asarray(density(grid), dtype=float), "density")
    values = np.broadcast_to(values, (grid_size,))
    integral = float(np.sum(values) * (b - a) / grid_size)
    if integral <= 0.0:
        raise ValueError("density has zero estimated mass")
    return lambda x: density(x) / integral, integral

def random_inverse_1d(
    density: Density1D,
    count: int,
    *,
    domain: tuple[float, float],
    grid_size: int = 1_000_000,
    rng: int | np.random.Generator | None = None,
) -> Array:
    """Sample a 1D density by grid-based inverse transform sampling.

    The density may be unnormalised. Returned particles have shape ``(N, 1)``.
    """
    rng = as_generator(rng)
    count = _validate_count(count)
    edges, cdf = _cdf_1d_from_density(density, domain=domain, grid_size=grid_size)
    quantiles = rng.uniform(size=count)
    return np.interp(quantiles, cdf, edges)[:, None]

def deterministic_quantile_1d(
    density: Density1D,
    count: int,
    *,
    domain: tuple[float, float],
    grid_size: int = 100_000,
) -> Array:
    """Construct deterministic midpoint quantiles for a 1D density.

    The density may be unnormalised. Returned particles have shape ``(N, 1)``.
    """
    count = _validate_count(count)
    edges, cdf = _cdf_1d_from_density(density, domain=domain, grid_size=grid_size)
    quantiles = (np.arange(count, dtype=float) + 0.5) / count
    return np.interp(quantiles, cdf, edges)[:, None]

def random_grid(
    density: DensityND,
    count: int,
    *,
    domain: Domain | tuple[float, float],
    grid_shape: int | Sequence[int] = 256,
    rng: int | np.random.Generator | None = None,
    jitter: bool = True,
) -> Array:
    """Sample an arbitrary rectangular-domain density by cell masses.

    The density may be unnormalised and should accept one coordinate array per
    dimension, using NumPy broadcasting. Returned particles have shape
    ``(N, d)``.
    """
    rng = as_generator(rng)
    count = _validate_count(count)
    domain_tuple = _coerce_domain(domain)
    dim = len(domain_tuple)
    grid_shape_tuple = _validate_grid_shape(grid_shape, dim)

    edges, centers, widths = _grid_geometry(domain_tuple, grid_shape_tuple)
    mesh = np.meshgrid(*centers, indexing="ij")
    weights = _positive_finite(np.asarray(density(*mesh), dtype=float), "density")
    if weights.shape != grid_shape_tuple:
        try:
            weights = np.broadcast_to(weights, grid_shape_tuple)
        except ValueError as exc:
            raise ValueError(
                "density output cannot be broadcast to grid_shape"
            ) from exc
    weights = weights.reshape(-1)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("density has zero estimated mass")
    probabilities = weights / total

    flat_indices = rng.choice(
        probabilities.size, size=count, replace=True, p=probabilities
    )
    cell_indices = np.column_stack(np.unravel_index(flat_indices, grid_shape_tuple))
    particles = np.empty((count, dim), dtype=float)
    for axis in range(dim):
        left = edges[axis][cell_indices[:, axis]]
        if jitter:
            particles[:, axis] = left + rng.uniform(size=count) * widths[axis]
        else:
            particles[:, axis] = centers[axis][cell_indices[:, axis]]
    return particles

def deterministic_rosenblatt_2d(
    density: Callable[[Array, Array], Array],
    count_per_axis: int,
    *,
    domain: Domain,
    grid_shape: int | Sequence[int] = 512,
) -> Array:
    """Construct 2D deterministic particles using a Rosenblatt transform.

    The density may be unnormalised. Returned particles have shape
    ``(count_per_axis**2, 2)``.
    """
    count_per_axis = _validate_count(count_per_axis, name="count_per_axis")
    domain_tuple = _validate_domain(domain, expected_dimension=2)
    nx, ny = _validate_grid_shape(grid_shape, 2)
    (x0, x1), (y0, y1) = domain_tuple

    x_edges = np.linspace(x0, x1, nx + 1)
    y_edges = np.linspace(y0, y1, ny + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny

    rho_grid = _positive_finite(
        np.asarray(density(x_centers[:, None], y_centers[None, :]), dtype=float),
        "density",
    )
    if rho_grid.shape != (nx, ny):
        try:
            rho_grid = np.broadcast_to(rho_grid, (nx, ny))
        except ValueError as exc:
            raise ValueError("density output cannot be broadcast to grid_shape") from exc

    marginal_mass = rho_grid.sum(axis=1) * dx * dy
    cdf_x = _cdf_from_cell_masses(marginal_mass)
    q = (np.arange(count_per_axis, dtype=float) + 0.5) / count_per_axis
    xs = np.interp(q, cdf_x, x_edges)

    particles = np.empty((count_per_axis * count_per_axis, 2), dtype=float)
    cursor = 0
    for xi in xs:
        conditional_mass = _positive_finite(
            np.asarray(density(xi, y_centers), dtype=float),
            "conditional density",
        )
        conditional_mass = np.broadcast_to(conditional_mass, (ny,)) * dy
        cdf_y = _cdf_from_cell_masses(conditional_mass)
        ys = np.interp(q, cdf_y, y_edges)
        particles[cursor : cursor + count_per_axis, 0] = xi
        particles[cursor : cursor + count_per_axis, 1] = ys
        cursor += count_per_axis
    return particles

def rejection_uniform(
    density: DensityND,
    count: int,
    *,
    domain: Domain | tuple[float, float],
    density_max: float | None = None,
    grid_shape: int | Sequence[int] = 128,
    safety_factor: float = 1.05,
    batch_size: int | None = None,
    rng: int | np.random.Generator | None = None,
    max_batches: int = 100_000,
) -> tuple[Array, float]:
    """Sample by rejection from a uniform proposal on a rectangular domain.

    If ``density_max`` is omitted, a grid estimate multiplied by
    ``safety_factor`` is used. Supplying an analytic upper bound is safer for
    production runs.
    """
    rng = as_generator(rng)
    count = _validate_count(count)
    domain_tuple = _coerce_domain(domain)
    dim = len(domain_tuple)
    if batch_size is None:
        batch_size = max(1024, 2 * count)
    batch_size = _validate_count(batch_size, name="batch_size")
    if max_batches <= 0:
        raise ValueError("max_batches must be positive")

    if density_max is None:
        density_max = estimate_density_max(
            density,
            domain=domain_tuple,
            grid_shape=grid_shape,
            safety_factor=safety_factor,
        )
    if not np.isfinite(density_max) or density_max <= 0.0:
        raise ValueError("density_max must be positive and finite")

    accepted: list[Array] = []
    accepted_count = 0
    total_draws = 0
    lows = np.array([interval[0] for interval in domain_tuple], dtype=float)
    highs = np.array([interval[1] for interval in domain_tuple], dtype=float)
    widths = highs - lows

    for _ in range(max_batches):
        proposal = lows + rng.uniform(size=(batch_size, dim)) * widths
        values = _evaluate_density_on_points(density, proposal)
        if np.any(values > density_max):
            raise ValueError(
                "encountered density value above density_max; increase the bound"
            )
        keep = rng.uniform(size=batch_size) <= values / density_max
        if np.any(keep):
            kept = proposal[keep]
            accepted.append(kept)
            accepted_count += kept.shape[0]
            if accepted_count >= count:
                total_draws += batch_size
                break
        total_draws += batch_size
    else:
        raise RuntimeError("rejection sampler did not collect enough particles")

    particles = np.vstack(accepted)[:count]
    return particles, count / total_draws

def estimate_density_max(
    density: DensityND,
    *,
    domain: Domain | tuple[float, float],
    grid_shape: int | Sequence[int] = 128,
    safety_factor: float = 1.05,
) -> float:
    """Estimate a rectangular-domain density maximum on a tensor grid."""
    if safety_factor < 1.0:
        raise ValueError("safety_factor must be at least one")
    domain_tuple = _coerce_domain(domain)
    dim = len(domain_tuple)
    grid_shape_tuple = _validate_grid_shape(grid_shape, dim)
    _, centers, _ = _grid_geometry(domain_tuple, grid_shape_tuple)
    mesh = np.meshgrid(*centers, indexing="ij")
    values = _positive_finite(np.asarray(density(*mesh), dtype=float), "density")
    if values.shape != grid_shape_tuple:
        values = np.broadcast_to(values, grid_shape_tuple)
    return float(np.max(values) * safety_factor)

def generate_initial_particles(
    *,
    method: ParticleMethod,
    count: int,
    seed: int | None = None,
    density: DensityND | None = None,
    density_1d: Density1D | None = None,
    density_2d: Callable[[Array, Array], Array] | None = None,
    domain: Domain | tuple[float, float] = ((0.0, 2.0 * np.pi),),
    grid_size: int | None = None,
    grid_shape: int | Sequence[int] | None = None,
    count_per_axis: int | None = None,
    density_max: float | None = None,
    safety_factor: float = 1.05,
    batch_size: int | None = None,
    jitter: bool = True,
) -> ParticleConfiguration:
    """Generate a reproducible initial particle configuration.

    Prefer this as the single entry point for both particle and SPDE initial
    data. Generate once, then pass ``config.particles`` to both simulations.
    """
    count = _validate_count(count)
    domain_tuple = _coerce_domain(domain)
    rng_seed = seed

    if method == "random-inverse-1d":
        density_fn = _select_density(density_1d, density, "density_1d")
        one_d_domain = _single_interval(domain_tuple)
        used_grid_size = grid_size or 1_000_000
        particles = random_inverse_1d(
            density_fn,
            count,
            domain=one_d_domain,
            grid_size=used_grid_size,
            rng=seed,
        )
        metadata = {"grid_size": used_grid_size}

    elif method == "deterministic-quantile-1d":
        density_fn = _select_density(density_1d, density, "density_1d")
        one_d_domain = _single_interval(domain_tuple)
        used_grid_size = grid_size or 100_000
        particles = deterministic_quantile_1d(
            density_fn,
            count,
            domain=one_d_domain,
            grid_size=used_grid_size,
        )
        metadata = {"grid_size": used_grid_size}

    elif method in {"random-grid", "random-grid-1d", "random-grid-2d"}:
        density_fn = _require_density(density, "density")
        if method == "random-grid-1d":
            _validate_domain(domain_tuple, expected_dimension=1)
        if method == "random-grid-2d":
            _validate_domain(domain_tuple, expected_dimension=2)
        used_grid_shape = grid_shape or 256
        particles = random_grid(
            density_fn,
            count,
            domain=domain_tuple,
            grid_shape=used_grid_shape,
            rng=seed,
            jitter=jitter,
        )
        metadata = {"grid_shape": _validate_grid_shape(used_grid_shape, len(domain_tuple)), "jitter": jitter}

    elif method == "deterministic-rosenblatt-2d":
        density_fn = _select_density(density_2d, density, "density_2d")
        domain_2d = _validate_domain(domain_tuple, expected_dimension=2)
        used_count_per_axis = (
            _validate_count(count_per_axis, name="count_per_axis")
            if count_per_axis is not None
            else int(np.ceil(np.sqrt(count)))
        )
        used_grid_shape = grid_shape or 512
        particles = deterministic_rosenblatt_2d(
            density_fn,
            used_count_per_axis,
            domain=domain_2d,
            grid_shape=used_grid_shape,
        )
        metadata = {
            "count_per_axis": used_count_per_axis,
            "grid_shape": _validate_grid_shape(used_grid_shape, 2),
        }

    elif method in {"rejection", "rejection-uniform"}:
        density_fn = _require_density(density, "density")
        used_grid_shape = grid_shape or 128
        used_density_max = density_max
        if used_density_max is None:
            used_density_max = estimate_density_max(
                density_fn,
                domain=domain_tuple,
                grid_shape=used_grid_shape,
                safety_factor=safety_factor,
            )
        particles, acceptance_rate = rejection_uniform(
            density_fn,
            count,
            domain=domain_tuple,
            density_max=used_density_max,
            grid_shape=used_grid_shape,
            safety_factor=safety_factor,
            batch_size=batch_size,
            rng=seed,
        )
        metadata = {
            "acceptance_rate": acceptance_rate,
            "density_max": used_density_max,
            "grid_shape": _validate_grid_shape(used_grid_shape, len(domain_tuple)),
            "safety_factor": safety_factor,
            "batch_size": batch_size,
        }

    else:
        raise ValueError(f"unknown particle generation method: {method}")

    return ParticleConfiguration(
        particles=particles,
        method=method,
        seed=rng_seed,
        requested_count=count,
        actual_count=int(particles.shape[0]),
        dimension=int(particles.shape[1]),
        domain=domain_tuple,
        metadata=metadata,
    )

def _cdf_1d_from_density(
    density: Density1D,
    *,
    domain: tuple[float, float],
    grid_size: int,
) -> tuple[Array, Array]:
    a, b = _validate_interval(domain)
    if grid_size <= 1:
        raise ValueError("grid_size must be greater than one")
    edges = np.linspace(a, b, grid_size + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    cell_width = (b - a) / grid_size
    cell_mass = _positive_finite(np.asarray(density(centers), dtype=float), "density")
    cell_mass = np.broadcast_to(cell_mass, (grid_size,)) * cell_width
    cdf = _cdf_from_cell_masses(cell_mass)
    return edges, cdf

def _cdf_from_cell_masses(cell_mass: Array) -> Array:
    cell_mass = _positive_finite(np.asarray(cell_mass, dtype=float), "cell_mass")
    total = float(np.sum(cell_mass))
    if total <= 0.0:
        raise ValueError("cell masses have zero total mass")
    cdf = np.concatenate(([0.0], np.cumsum(cell_mass) / total))
    cdf[-1] = 1.0
    return cdf

def _grid_geometry(
    domain: tuple[tuple[float, float], ...],
    grid_shape: tuple[int, ...],
) -> tuple[tuple[Array, ...], tuple[Array, ...], tuple[float, ...]]:
    edges = []
    centers = []
    widths = []
    for (a, b), n in zip(domain, grid_shape, strict=True):
        axis_edges = np.linspace(a, b, n + 1)
        edges.append(axis_edges)
        centers.append(0.5 * (axis_edges[:-1] + axis_edges[1:]))
        widths.append((b - a) / n)
    return tuple(edges), tuple(centers), tuple(widths)

def _evaluate_density_on_points(density: DensityND, points: Array) -> Array:
    coords = [points[:, axis] for axis in range(points.shape[1])]
    values = np.asarray(density(*coords), dtype=float)
    values = np.broadcast_to(values, (points.shape[0],))
    return _positive_finite(values, "density")


def _positive_finite(values: Array, name: str) -> Array:
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(values < 0.0):
        raise ValueError(f"{name} contains negative values")
    return values


def _validate_count(count: int | None, name: str = "count") -> int:
    if count is None:
        raise ValueError(f"{name} is required")
    count = int(count)
    if count <= 0:
        raise ValueError(f"{name} must be positive")
    return count

def _validate_interval(interval: tuple[float, float]) -> tuple[float, float]:
    a, b = float(interval[0]), float(interval[1])
    if not np.isfinite(a) or not np.isfinite(b) or not a < b:
        raise ValueError("domain intervals must be finite and increasing")
    return a, b


def _coerce_domain(
    domain: Domain | tuple[float, float],
) -> tuple[tuple[float, float], ...]:
    if len(domain) == 2 and all(np.isscalar(value) for value in domain):  # type: ignore[arg-type]
        return (_validate_interval(domain),)  # type: ignore[arg-type]
    return _validate_domain(domain)  # type: ignore[arg-type]


def _validate_domain(
    domain: Domain,
    expected_dimension: int | None = None,
) -> tuple[tuple[float, float], ...]:
    domain_tuple = tuple(_validate_interval(interval) for interval in domain)
    if not domain_tuple:
        raise ValueError("domain must contain at least one interval")
    if expected_dimension is not None and len(domain_tuple) != expected_dimension:
        raise ValueError(f"expected a {expected_dimension}D domain")
    return domain_tuple


def _single_interval(domain: tuple[tuple[float, float], ...]) -> tuple[float, float]:
    domain = _validate_domain(domain, expected_dimension=1)
    return domain[0]

def _validate_grid_shape(
    grid_shape: int | Sequence[int],
    dimension: int,
) -> tuple[int, ...]:
    if isinstance(grid_shape, int):
        shape = (int(grid_shape),) * dimension
    else:
        shape = tuple(int(value) for value in grid_shape)
    if len(shape) != dimension:
        raise ValueError("grid_shape length must match domain dimension")
    if any(value <= 0 for value in shape):
        raise ValueError("grid_shape entries must be positive")
    return shape


def _require_density(density: Callable | None, name: str) -> Callable:
    if density is None:
        raise ValueError(f"{name} is required for this method")
    return density


def _select_density(
    primary: Callable | None, fallback: Callable | None, name: str
) -> Callable:
    if primary is not None:
        return primary
    return _require_density(fallback, name)