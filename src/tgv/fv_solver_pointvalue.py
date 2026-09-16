"""
A7 ablation (constitution v2 §10, §6 leakage risk): a POINT-VALUE /
advective-form discretization, identical order of accuracy to
fv_solver.py's flux-difference form, but computing the nonlinear
convective term as u*du/dx directly at cell centers rather than as a
difference of shared face fluxes u_face^2.

The two forms are equal in the continuum (product rule) but differ
discretely. Flux form conserves the domain-integrated momentum EXACTLY
(the face-flux differences telescope to zero when summed over a periodic
domain, regardless of the flux's value) -- point-value form has no such
algebraic guarantee. This is the concrete, measurable distinction A7
uses to prove our solver is genuinely flux-form, not point-value FD
wearing an FV label.
"""
import numpy as np

from .fv_solver import project_divergence_free, divergence


def convection_diffusion_rhs_pointvalue(u, v, dx, dy, nu):
    """Advective (non-conservative) form: u*du/dx + v*du/dy for the
    u-momentum equation (and analogously for v), all via central
    differences evaluated directly at cell centers -- no shared face
    values, no flux telescoping."""
    dudx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dx)
    dudy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dy)
    dvdx = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * dx)
    dvdy = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2 * dy)

    lap_u = (np.roll(u, -1, axis=0) - 2 * u + np.roll(u, 1, axis=0)) / dx**2 \
        + (np.roll(u, -1, axis=1) - 2 * u + np.roll(u, 1, axis=1)) / dy**2
    lap_v = (np.roll(v, -1, axis=0) - 2 * v + np.roll(v, 1, axis=0)) / dx**2 \
        + (np.roll(v, -1, axis=1) - 2 * v + np.roll(v, 1, axis=1)) / dy**2

    rhs_u = -(u * dudx + v * dudy) + nu * lap_u
    rhs_v = -(u * dvdx + v * dvdy) + nu * lap_v
    return rhs_u, rhs_v


def run_pointvalue(u0, v0, dx, dy, nu, dt, n_steps, rho=1.0):
    """Same RK2 + spectral-projection time integration as fv_solver.run,
    but using the point-value/advective RHS above -- isolates the
    conservative-vs-non-conservative discretization choice as the only
    difference between the two arms of the A7 ablation."""
    u, v = u0.copy(), v0.copy()
    p = None
    total_u_history = []
    for _ in range(n_steps):
        ru0, rv0 = convection_diffusion_rhs_pointvalue(u, v, dx, dy, nu)
        u_pred = u + dt * ru0
        v_pred = v + dt * rv0

        ru1, rv1 = convection_diffusion_rhs_pointvalue(u_pred, v_pred, dx, dy, nu)
        u_star = u + 0.5 * dt * (ru0 + ru1)
        v_star = v + 0.5 * dt * (rv0 + rv1)

        u, v, p = project_divergence_free(u_star, v_star, dx, dy, dt, rho)
        total_u_history.append(float(np.sum(u)))

    return u, v, p, total_u_history
