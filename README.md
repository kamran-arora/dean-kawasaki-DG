# A discontinuous Galerkin approximation of the Dean-Kawasaki equation

Reference implementation for the paper *A discontinuous Galerkin approximation of the Dean-Kawasaki equation* ([arXiv](/)).

`dean-kawasaki-DG` is the library; `experiments/` contains the scripts that reproduce the results in the paper.

The code is written using the [Firedrake](https://github.com/firedrakeproject/firedrake) finite element library.

## Reproducing figures

The following require you to first run [dk_monte_carlo.py](experiments/dk_monte_carlo.py) using the parameters specified in the paper
- Figure 1: [smooth_varphi.ipynb](experiments/smooth_varphi.ipynb)
- Figure 2a: [indicator_varphi.ipynb](experiments/indicator_varphi.ipynb)
- Figure 2b: [indicator_varphi.ipynb](experiments/indicator_varphi.ipynb)

The following can be run as standalone notebooks
- Figure 2a: [external_potential.ipynb](experiments/external_potential.ipynb)
- Figure 2b: [external_potential.ipynb](experiments/external_potential.ipynb)
- Figure 3: [interaction_potential.ipynb](experiments/interaction_potential.ipynb)
- Figure 4: [RBM.ipynb](experiments/RBM.ipynb) (requires gmsh)

## Future development

- [ ] integrate external potentials and interaction potentials into the solver API
- [ ] integrate a zero-flux Neumann boundary condition into the solver API