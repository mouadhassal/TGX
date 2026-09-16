"""
K1 knob, grounded in the actual FV solver (constitution v2 §10, A1/A7
analogue): "error and rank flat in Re at K1=0, both jumping
discontinuously as soon as K1 > 0" -- turn the degeneracy claim into a
measurement, using the same solver already validated for G0-G2.

We run the UNFORCED solver (no MMS source term) starting from the
K1-perturbed IC at eps in {0, 0.1, 0.3, 0.6}. At eps=0 the IC is the
exact unforced TGV solution (D1), so the solver's error against the
(still-exact, still eps=0) analytic solution stays at pure
discretization-error level. At eps>0 the IC is no longer an exact
solution of the UNFORCED equations (see test_d1_k1_perturbed_field_
breaks_the_cancellation: forcing != 0 for eps != 0) -- so the unforced
solver increasingly mismatches its own (now missing-forcing) exact
reference, and the field's own quantics rank should grow, both
discontinuously relative to the eps=0 corner.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from tgv.fv_solver import run
from tgv.mms import k1_perturbed_field
from tgv.quantics import quantics_reshape_2d, tt_ranks
from tgv.norms import relative_l2_velocity


def _grid(N, domain_length=2 * np.pi):
    dx = domain_length / N
    xs = (np.arange(N) + 0.5) * dx
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    return X, Y, dx


def test_k1_unforced_solver_error_flat_at_zero_then_jumps():
    """The UNFORCED solver only ever sees a solution of the unforced
    equations at eps=0 (D1); for eps>0 it's missing K1's forcing term by
    construction, so its error against the true (still eps>0) exact
    solution should jump well above the eps=0 discretization floor."""
    N = 64
    X, Y, dx = _grid(N)
    nu = 0.05
    T = 0.5

    errs = []
    for eps in (0.0, 0.1, 0.3, 0.6):
        field = k1_perturbed_field(eps=eps, k2=3, Uc=1.0, Vc=0.0, V0=1.0, L=1.0, p0=0.0)
        u_num, v_num, _, _, _ = field.lambdify_all(nu_val=nu)
        u0 = u_num(X, Y, 0.0) * np.ones_like(X)
        v0 = v_num(X, Y, 0.0) * np.ones_like(X)

        dt = 0.2 * dx**2 / (4 * nu)
        n_steps = max(1, int(round(T / dt)))
        dt = T / n_steps

        u, v, press, _ = run(u0, v0, dx, dx, nu, dt, n_steps=n_steps)

        u_ex = u_num(X, Y, T) * np.ones_like(X)
        v_ex = v_num(X, Y, T) * np.ones_like(X)
        err = relative_l2_velocity(u, v, u_ex, v_ex)
        errs.append((eps, err))

    for eps, err in errs:
        print(f"eps={eps}: err={err:.3e}", file=sys.stderr)

    err0 = errs[0][1]
    err_max = errs[-1][1]

    assert err0 < 5e-3, errs             # flat: pure discretization floor
    assert err_max > 50 * err0, errs     # jump: far above that floor


def test_k1_exact_field_rank_grows_with_eps_at_fixed_tight_tolerance():
    """Separate from solver discretization noise: the EXACT K1-perturbed
    field's own TT-rank (measured on the analytic field itself, tight,
    fixed tolerance -- no discretization error to worry about) must be
    monotonically non-decreasing as eps grows from 0, since a second
    incommensurate mode is being added to the base rank<=5 field."""
    N = 64
    X, Y, dx = _grid(N)
    nu = 0.05
    T = 0.5

    chis = []
    for eps in (0.0, 0.1, 0.3, 0.6):
        field = k1_perturbed_field(eps=eps, k2=3, Uc=1.0, Vc=0.0, V0=1.0, L=1.0, p0=0.0)
        u_num, _, _, _, _ = field.lambdify_all(nu_val=nu)
        u_ex = u_num(X, Y, T) * np.ones_like(X)
        chi = max(tt_ranks(quantics_reshape_2d(u_ex), tol=1e-9))
        chis.append((eps, chi))

    for eps, chi in chis:
        print(f"eps={eps}: chi(exact field)={chi}", file=sys.stderr)

    chi0 = chis[0][1]
    assert chi0 <= 5, chis                       # eps=0: matches D3's bound
    assert chis[-1][1] >= chi0, chis             # non-decreasing with eps
    assert max(c for _, c in chis[1:]) > chi0, chis  # strictly grows somewhere
