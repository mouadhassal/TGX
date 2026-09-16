"""
A7 (constitution v2 §10 / §6 leakage risk, the most embarrassing possible
failure mode named in v1's red team): prove, don't assert, that our
solver is genuinely flux-form by showing it conserves domain-integrated
momentum to machine precision while an otherwise-identical point-value
(advective-form) discretization does not.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from tgv.fv_solver import step
from tgv.fv_solver_pointvalue import run_pointvalue
from tgv.config import TGVParams
from tgv.analytic import velocity
from tgv.mms import base_tgv_field, k1_perturbed_field


def _grid(N, domain_length=2 * np.pi):
    dx = domain_length / N
    xs = (np.arange(N) + 0.5) * dx
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    return X, Y, dx


def test_a7_flux_form_conserves_momentum_to_machine_precision():
    """Even for the base (linear-on-trajectory, per D1) TGV case, flux
    form must conserve sum(u) exactly over many steps."""
    N = 32
    X, Y, dx = _grid(N)
    p = TGVParams(Re=100)
    u, v = velocity(p, X, Y, 0.0)

    total_u0 = np.sum(u)
    dt = 0.1 * dx / abs(p.Uc)

    for _ in range(50):
        u, v, press, diag = step(u, v, dt, dx, dx, p.nu)

    drift = abs(np.sum(u) - total_u0)
    assert drift < 1e-8 * abs(total_u0), drift


def test_a7_pointvalue_form_breaks_conservation_under_k1_perturbation():
    """The base TGV alone is a weak test (D1 makes the nonlinear term
    nearly vanish at the grid scale). Use the K1-perturbed field (genuine
    two-mode nonlinearity, eps != 0) as initial condition -- unforced, so
    both arms integrate the SAME unforced NS equations from the SAME IC,
    and only the discretization's conservation property differs."""
    N = 32
    X, Y, dx = _grid(N)
    nu = 0.05
    eps = 0.5

    field = k1_perturbed_field(eps=eps, k2=3, Uc=1.0, Vc=0.0, V0=1.0, L=1.0, p0=0.0)
    u_num, v_num, _, _, _ = field.lambdify_all(nu_val=nu)
    u0 = u_num(X, Y, 0.0) * np.ones_like(X)
    v0 = v_num(X, Y, 0.0) * np.ones_like(X)

    dt = 0.05 * dx
    n_steps = 100

    # flux-form arm (unforced -- ignores the K1 forcing term deliberately,
    # since we only care about each scheme's OWN conservation property
    # under its own (possibly slightly wrong) unforced evolution)
    u_flux, v_flux = u0.copy(), v0.copy()
    total_u0 = np.sum(u_flux)
    for _ in range(n_steps):
        u_flux, v_flux, _, _ = step(u_flux, v_flux, dt, dx, dx, nu)
    drift_flux = abs(np.sum(u_flux) - total_u0)

    # point-value arm
    u_pv, v_pv, _, history = run_pointvalue(u0, v0, dx, dx, nu, dt, n_steps)
    drift_pointvalue = abs(history[-1] - total_u0)

    assert drift_flux < 1e-8 * abs(total_u0), drift_flux
    # the point-value arm has no algebraic reason to conserve momentum;
    # its drift must be many orders of magnitude larger than the flux
    # arm's machine-precision drift
    assert drift_pointvalue > 1e4 * max(drift_flux, 1e-300), (drift_flux, drift_pointvalue)
