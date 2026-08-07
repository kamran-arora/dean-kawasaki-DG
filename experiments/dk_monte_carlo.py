import numpy as np
from timeit import default_timer as timer
from functools import partial
import multiprocessing as mp
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _firedrake_argv(argv):
    petsc_argv = [argv[0]]
    i = 1
    while i < len(argv):
        if argv[i] == "-log_view":
            petsc_argv.append(argv[i])
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                petsc_argv.append(argv[i + 1])
                i += 1
        i += 1
    return petsc_argv


_argv = sys.argv[:]
sys.argv = _firedrake_argv(sys.argv)
try:
    from firedrake import *
finally:
    sys.argv = _argv
    del _argv

import dean_kawasaki_DG as dkDG

from dean_kawasaki_DG import generate_initial_particles


LENGTH = 2 * np.pi


def _format_parameter(value):
    return f"{value:g}"


def build_experiment_id(
    params: dict,
    total_runs: int,
    procs: int,
    SEED: int,
    *,
    created_at: datetime | None = None,
):
    created_at = created_at or datetime.now().astimezone()
    timestamp = created_at.strftime("%Y%m%dT%H%M%S%f")
    return "_".join(
        [
            timestamp,
            f"p{params['degree']}",
            f"nx{params['nx']:03d}",
            f"N{params['N']}",
            f"dt{_format_parameter(params['dt'])}",
            f"T{_format_parameter(params['T'])}",
            f"kappa{_format_parameter(params['kappa'])}",
            f"eta{_format_parameter(params['eta'])}",
            params["variant"],
            params["initial_condition_method"],
            params["noise_gradient"],
            f"seed{SEED}",
            f"M{total_runs}",
            f"workers{procs}",
        ]
    )


def artifact_path(fname: str, kind: str, params: dict, extension: str):
    return (
        f"{fname}_{kind}_p{params['degree']}_nx{params['nx']:03d}"
        f"_noise-{params['noise_gradient']}.{extension}"
    )


def build_run_params(
    nx: int,
    N: int,
    eta: float,
    variant: str,
    initial_condition_method: str,
    noise_gradient: str,
):
    return {
        "nx": nx,
        "N": N,
        "eta": eta,
        "dt": 0.001,
        "T": 0.1,
        "kappa": 1.0,
        "degree": 2,
        "variant": variant,
        "initial_condition_method": initial_condition_method,
        "noise_gradient": noise_gradient,
    }

def rho_init(x):
    return (
        1
        + 0.35 * np.sin(2 * x + 0.4)
        + 0.25 * np.cos(3 * x - 0.7)
    ) / (2 * np.pi)


def rho_init_ufl(x):
    return (
        1
        + 0.35 * sin(2 * x + 0.4)
        + 0.25 * cos(3 * x - 0.7)
    ) / (2 * np.pi)


def build_initial_particle_configuration(
    method: str, N: int, SEED: int, grid_size: int = 10**6
):
    """
    Generate the particle configuration associated to an initial condition construction.

    :method: either "particle-first" or "mfl-first"
    :N: number of particles
    :SEED: random seed
    :grid_size: grid size used by the 1D particle generator
    """

    particle_method = {
        "particle-first": "random-inverse-1d",
        "mfl-first": "deterministic-quantile-1d",
    }[method]
    return generate_initial_particles(
        method=particle_method,
        count=N,
        seed=SEED,
        density_1d=rho_init,
        domain=(0.0, LENGTH),
        grid_size=grid_size,
    )


def build_initial_particles(method: str, N: int, SEED: int, grid_size: int = 10**6):
    """
    Generate the particles associated to an initial condition construction.

    :method: either "particle-first" or "mfl-first"
    :N: number of particles
    :SEED: random seed
    :grid_size: grid size used by the 1D particle generator
    """

    config = build_initial_particle_configuration(method, N, SEED, grid_size)
    return config.particles


def save_particle_configuration(config, FOLDER_PATH: str):
    """
    Save the particle positions and metadata used to build the initial condition.

    :config: particle configuration
    :FOLDER_PATH: folder to save data in
    """

    np.save(
        FOLDER_PATH + "particle_configuration.npy",
        config.particles,
        allow_pickle=False,
    )
    metadata = {
        "particles_file": "particle_configuration.npy",
        "method": config.method,
        "seed": config.seed,
        "requested_count": config.requested_count,
        "actual_count": config.actual_count,
        "dimension": config.dimension,
        "domain": config.domain,
        "metadata": config.metadata,
    }
    with open(FOLDER_PATH + "particle_configuration.json", "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")


def set_initial_condition(model, method: str, particles, x):
    """
    Set the initial condition using the requested construction method.

    :model: heat or Dean-Kawasaki solver
    :method: either "particle-first" or "mfl-first"
    :particles: particle positions
    :x: spatial coordinate
    """

    if method == "particle-first":
        model.initial_condition_from_particles(particles)
    elif method == "mfl-first":
        model.set_initial_condition(rho_init_ufl(x))
    else:
        raise ValueError("initial condition method must be 'particle-first' or 'mfl-first'")


def multi_solve(runs: int, fname: str, SEED: int, params: dict):
    """
    Solves the heat equation and multiple realisations of Dean-Kawasaki, saving raw sample path data at the final time

    :runs: number of realisations to solve
    :fname:
    :params: dictionary of problem parameters
    :SEED: the SEED passed to the Generator object in the DK class
    """

    # extract parameters
    nx = params["nx"]
    dt = params["dt"]
    T = params["T"]
    eta = params["eta"]
    N = params["N"]
    kappa = params["kappa"]
    degree = params["degree"]
    particles = params["particles"]
    variant = params["variant"]
    initial_condition_method = params["initial_condition_method"]
    noise_gradient = params["noise_gradient"]

    # setup mesh
    mesh = PeriodicIntervalMesh(ncells=nx, length=LENGTH)
    (x,) = SpatialCoordinate(mesh)

    # setup dk problem
    args = (kappa, N, eta)
    model_dk = dkDG.DK_SIPDG_BE_1D(
        mesh,
        int(degree),
        dt,
        args,
        variant,
        SEED=SEED,
        noise_gradient=noise_gradient,
        noise_sampling_backend="cell",
    )
    set_initial_condition(model_dk, initial_condition_method, particles, x)

    # setup mfl problem
    args = (kappa, eta)
    model_mfl = dkDG.HEAT_SIPDG_BE_1D(mesh, int(degree), dt, args, variant)
    set_initial_condition(model_mfl, initial_condition_method, particles, x)

    # setup exact by solving mfl and solve
    exact = model_mfl.solve(T, None)
    np.save(
        artifact_path(fname, "mfl", params, "npy"),
        exact.dat.data_ro,
        allow_pickle=False,
    )

    # storage
    all_rhos_data_ro = []
    # loop
    for _ in range(len(runs)):
        # solve and store
        rho = model_dk.solve(T, None)
        all_rhos_data_ro.append(rho.dat.data_ro.copy())
    # save to file
    np.savez(
        artifact_path(fname, "dk", params, "npz"),
        *all_rhos_data_ro,
        allow_pickle=False,
    )


def main(
    procs: int,
    total_runs: int,
    nx: int,
    N: int,
    eta: float,
    SEED: int,
    FOLDER_PATH: str,
    variant: str,
    initial_condition_method: str,
    noise_gradient: str,
):
    """
    Parallelise DK solves over processors

    :procs: number of processors to parallelise over
    :total_runs: total realisations to simulate
    :nx: mesh width
    :N: number of particles
    :eta: SIPDG penalty parameter
    :SEED: master seed
    :variant: Firedrake element variant
    :initial_condition_method: either "particle-first" or "mfl-first"
    :noise_gradient: either "broken" or "full"

        We use this seed to generate the particle initial positions. We then spawn child seeds from this corresponding to the number of processors we wish to parallelise over. These are the seeds passed to each invocation of multi_solve.

    """

    # common parameters
    params = build_run_params(
        nx,
        N,
        eta,
        variant,
        initial_condition_method,
        noise_gradient,
    )

    # particles
    np.random.seed(SEED)
    particle_config = build_initial_particle_configuration(
        initial_condition_method, N, SEED
    )
    params["particles"] = particle_config.particles
    save_particle_configuration(particle_config, FOLDER_PATH)

    # solve once so firedrake caches stuff
    multi_solve(
        np.arange(1), fname=FOLDER_PATH + "warmup", SEED=SEED, params=params
    )
    # split total runs over processors
    split_runs = np.array_split(np.linspace(0, 1, total_runs), procs)
    # create file names
    fnames = [FOLDER_PATH + f"worker{i:02d}" for i in range(procs)]
    # create a partial function to fix params
    partial_multi_solve = partial(multi_solve, params=params)
    # generate a set of child seeds
    children = np.random.SeedSequence(SEED).spawn(procs)
    child_seeds = [s.generate_state(1)[0] for s in children]
    # run in parallel
    with mp.Pool(procs) as pool:
        # we loop over tuples of (number of runs, filename, child_seed)
        pool.starmap(partial_multi_solve, zip(split_runs, fnames, child_seeds))
    return params


if __name__ == "__main__":
    # get nx and total_runs as command line arguments
    parser = argparse.ArgumentParser(description="DK SIPDG Monte-Carlo")
    parser.add_argument("nx", type=int, help="mesh size")
    parser.add_argument("total_runs", type=int, help="total runs")
    parser.add_argument("N", type=int, help="number of particles")
    parser.add_argument("eta", type=float, help="penalty parameter")
    parser.add_argument("SEED", type=int, help="random seed")
    parser.add_argument(
        "--variant",
        default="equispaced",
        help="Firedrake element variant",
    )
    parser.add_argument(
        "--initial-condition-method",
        "--ic-method",
        choices=["particle-first", "mfl-first"],
        default="particle-first",
        help="initial condition construction method",
    )
    parser.add_argument(
        "--noise-gradient",
        "--noise-type",
        choices=["broken", "full"],
        default="broken",
        help="Dean-Kawasaki noise gradient type",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="parent directory for uniquely named experiment folders",
    )
    cmd_args, _petsc_args = parser.parse_known_args()
    nx = cmd_args.nx
    total_runs = cmd_args.total_runs
    N = cmd_args.N
    eta = cmd_args.eta
    SEED = cmd_args.SEED
    variant = cmd_args.variant
    initial_condition_method = cmd_args.initial_condition_method
    noise_gradient = cmd_args.noise_gradient
    output_root = cmd_args.output_root
    # get number of processors
    #procs = mp.cpu_count()
    procs = 5

    # make a unique folder to store data in
    created_at = datetime.now().astimezone()
    params = build_run_params(
        nx,
        N,
        eta,
        variant,
        initial_condition_method,
        noise_gradient,
    )
    experiment_id = build_experiment_id(
        params,
        total_runs,
        procs,
        SEED,
        created_at=created_at,
    )
    full_path = Path(output_root) / experiment_id
    full_path.mkdir(parents=True, exist_ok=False)
    FOLDER_PATH = str(full_path) + os.sep

    # run experiment
    t0 = timer()
    params = main(
        procs,
        total_runs,
        nx,
        N,
        eta,
        SEED,
        FOLDER_PATH,
        variant,
        initial_condition_method,
        noise_gradient,
    )
    t1 = timer()

    # save cmd args into a txt file
    output_lines = [
        f"experiment ID = {experiment_id}",
        "-----------------",
        f"creation timestamp = {created_at.isoformat()}",
        "-----------------",
        f"procs = {procs}",
        "-----------------",
        f"nx = {nx}",
        "-----------------",
        f"N = {N}",
        "-----------------",
        f"total runs = {total_runs}",
        "-----------------",
        f"eta = {eta}",
        "-----------------",
        f"dt = {params['dt']}",
        "-----------------",
        f"T = {params['T']}",
        "-----------------",
        f"kappa = {params['kappa']}",
        "-----------------",
        f"polynomial degree = {params['degree']}",
        "-----------------",
        f"SEED = {SEED}",
        "-----------------",
        f"variant = {variant}",
        "-----------------",
        f"initial condition method = {initial_condition_method}",
        "-----------------",
        f"noise gradient = {noise_gradient}",
        "-----------------",
        f"Total time = {t1 - t0}",
        "------------------",
    ]
    with open(FOLDER_PATH + "params.txt", "w") as f:
        for line in output_lines:
            f.write(line + "\n")
