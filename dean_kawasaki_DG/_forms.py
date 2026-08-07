from firedrake import *


def cg_diffusion_form(u, v):
    """
    Continuous Galerkin diffusion bilinear form.

    Args:
    - u: firedrake.TrialFunction
    - v: firedrake.TestFunction
    """
    return inner(grad(u), grad(v)) * dx


def sipdg_diffusion_form(mesh, u, v, eta):
    """
    Symmetric interior penalty dG diffusion bilinear form with penalty
    parameter eta > 0

    Args:
    - u: firedrake.TrialFunction
    - v: firedrake.TestFunction
    - eta: float
    """
    n = FacetNormal(mesh)
    hh = CellSize(mesh)
    h = (hh("+") + hh("-")) / 2.0

    cell_term = inner(grad(u), grad(v)) * dx
    consistency = inner(avg(grad(u)), jump(v, n)) * dS
    symmetry = inner(avg(grad(v)), jump(u, n)) * dS
    penalty = eta / h * inner(jump(u, n), jump(v, n)) * dS
    return cell_term - consistency - symmetry + penalty
