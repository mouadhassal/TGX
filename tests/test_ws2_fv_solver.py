"""
WS2 / G2: dense flux-form FV solver verification.
  1. Divergence-free to machine precision on every step (§9.4 invariant).
  2. MMS grid-convergence study: observed order >= 1.8 (G2 threshold).
  3. Pure TGV (unforced, D1) stays close to the exact solution and its
     error shrinks under grid refinement -- sanity check before trusting
     any scaling study built on this solver.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import sympy as sp

from tgv.fv_solver import run, divergence
from tgv.mms import base_tgv_field, k1_perturbed_field
from tgv.norms import relative_l2_velocity


def _grid(N, domain_length=2 * np.pi):
    dx = domain_length / N
    xs = (np.arange(N) + 0.5) * dx  # cell centers
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    return X, Y, dx


def test_ws2_pure_tgv_stays_divergence_free_to_machine_precision():
    N = 32
    X, Y, dx = _grid(N)
    Uc, nu = 1.0, 0.05

    u_num, v_num, _, _, _ = base_tgv_field().lambdify_all(nu_val=nu)
    u0 = u_num(X, Y, 0.0) * np.ones_like(X)
    v0 = v_num(X, Y, 0.0) * np.ones_like(X)

    dt = 0.2 * dx / max(abs(Uc), 1e-9)
    u, v, p, diags = run(u0, v0, dx, dx, nu, dt, n_steps=20)

    max_div = max(d.max_div for d in diags)
    assert max_div < 1e-9, max_div


def test_ws2_pure_tgv_error_shrinks_under_refinement():
    """Unforced D1 case: no MMS forcing needed since the exact solution
    already solves the equations; error should be small and decrease as
    N increases (spatial discretization error only)."""
    Uc, nu, T = 1.0, 0.05, 0.05
    errors = []
    for N in (16, 32, 64):
        X, Y, dx = _grid(N)
        u_num, v_num, _, _, _ = base_tgv_field().lambdify_all(nu_val=nu)
        u0 = u_num(X, Y, 0.0) * np.ones_like(X)
        v0 = v_num(X, Y, 0.0) * np.ones_like(X)

        dt = 0.1 * dx / abs(Uc)
        n_steps = max(1, int(round(T / dt)))
        dt = T / n_steps

        u, v, p, _ = run(u0, v0, dx, dx, nu, dt, n_steps=n_steps)

        u_ex = u_num(X, Y, T) * np.ones_like(X)
        v_ex = v_num(X, Y, T) * np.ones_like(X)
        err = relative_l2_velocity(u, v, u_ex, v_ex)
        errors.append(err)

    assert errors[-1] < errors[0]
    assert errors[1] < errors[0]


def test_g2_mms_convergence_order_at_least_1_8():
    """K1-perturbed manufactured field (nonzero forcing, genuinely
    exercises the nonlinear term): grid-refinement study of relative L2
    error at fixed final time, observed order must be >= 1.8 (G2)."""
    eps = 0.3
    nu = 0.05
    field = k1_perturbed_field(eps=eps, k2=3, Uc=1.0, Vc=0.0, V0=1.0, L=1.0, p0=0.0)
    u_num, v_num, p_num, Fx_num, Fy_num = field.lambdify_all(nu_val=nu)

    def forcing_factory(X, Y):
        def forcing(t):
            Fx = np.asarray(Fx_num(X, Y, t), dtype=float) * np.ones_like(X)
            Fy = np.asarray(Fy_num(X, Y, t), dtype=float) * np.ones_like(X)
            return Fx, Fy
        return forcing

    T = 0.02
    Ns = (16, 32, 64)
    errors = []
    for N in Ns:
        X, Y, dx = _grid(N)
        u0 = u_num(X, Y, 0.0) * np.ones_like(X)
        v0 = v_num(X, Y, 0.0) * np.ones_like(X)

        dt = 0.05 * dx  # small enough that temporal error doesn't dominate
        n_steps = max(1, int(round(T / dt)))
        dt = T / n_steps

        u, v, p, _ = run(u0, v0, dx, dx, nu, dt, n_steps=n_steps,
                          forcing=forcing_factory(X, Y), t0=0.0)

        u_ex = u_num(X, Y, T) * np.ones_like(X)
        v_ex = v_num(X, Y, T) * np.ones_like(X)
        err = relative_l2_velocity(u, v, u_ex, v_ex)
        errors.append(err)

    log_dx = np.log([2 * np.pi / N for N in Ns])
    log_err = np.log(errors)
    order = np.polyfit(log_dx, log_err, 1)[0]

    assert order >= 1.8, (errors, order)
