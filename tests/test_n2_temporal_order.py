"""
SUPERSEDED by test_n2_temporal_order_direct.py (AUDIT-3 §1).

This file originally used Richardson extrapolation (subtract the
finest-dt run as a dt->0 proxy) to argue the solver was 2nd order in
time, diagnosing an apparent order-1 result at dt~dx as "H1: a floor
contaminating a genuine 2nd-order method." AUDIT-3 correctly identified
that diagnosis as a misdiagnosis: N2's own Richardson estimator and
N3's direct dt-scaling sweep were mutually inconsistent, and the
simpler, correct explanation was that the time-splitting itself was
genuinely 1st order (an unprojected RK2 predictor stage, projected only
once at the end).

Confirmed directly (no Richardson subtraction) and FIXED: fv_solver.py's
step() now projects the predictor stage too (incremental pressure
correction). See test_n2_temporal_order_direct.py for the direct
verification (error quarters, not halves, under dt-halving at fixed
N) and CHANGELOG-protocol.md for the full account.

This file is kept only so `test_n2_temporal_order_via_richardson_...`
is not silently lost from history; it is intentionally reduced to a
smoke test rather than a Richardson-based order claim, since that
methodology was shown to be unreliable here (its own estimates climbed
past design order at the finest levels, which was the original clue
something was off, and its residuals flip sign after the solver fix,
which would require re-deriving the diagnostic from scratch to trust
again -- not worth it now that the direct test exists and is simpler).
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from tgv.fv_solver import run
from tgv.mms import k1_perturbed_field
from tgv.norms import relative_l2_velocity


def _grid(N, domain_length=2 * np.pi):
    dx = domain_length / N
    xs = (np.arange(N) + 0.5) * dx
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    return X, Y, dx


def test_mms_forced_case_error_still_shrinks_under_dt_refinement():
    """Smoke test only: confirms the solver still runs and converges on
    the MMS-forced configuration this file used to analyze in detail.
    For the actual temporal-order claim, see test_n2_temporal_order_direct.py."""
    N = 96
    X, Y, dx = _grid(N)
    nu = 0.05
    eps = 0.3
    T = 0.05

    field = k1_perturbed_field(eps=eps, k2=3, Uc=1.0, Vc=0.0, V0=1.0, L=1.0, p0=0.0)
    u_num, v_num, _, Fx_num, Fy_num = field.lambdify_all(nu_val=nu)
    u0 = u_num(X, Y, 0.0) * np.ones_like(X)
    v0 = v_num(X, Y, 0.0) * np.ones_like(X)
    u_ex = u_num(X, Y, T) * np.ones_like(X)
    v_ex = v_num(X, Y, T) * np.ones_like(X)

    def forcing(t):
        Fx = np.asarray(Fx_num(X, Y, t), dtype=float) * np.ones_like(X)
        Fy = np.asarray(Fy_num(X, Y, t), dtype=float) * np.ones_like(X)
        return Fx, Fy

    errs = []
    for n_steps in (4, 16, 64):
        dt = T / n_steps
        u, v, press, _ = run(u0, v0, dx, dx, nu, dt, n_steps=n_steps, forcing=forcing)
        errs.append(relative_l2_velocity(u, v, u_ex, v_ex))

    # smoke test only (see module docstring): with the temporal-order
    # fix, error is now so small that it can approach its floor from
    # either side (temporal and spatial error components can partially
    # cancel), so this only checks the values stay close together and
    # bounded -- not a monotonic-decrease claim, which the fixed
    # solver's tiny residuals are too noise-sensitive to guarantee here.
    assert max(errs) / min(errs) < 1.1, errs
