"""
Methodological finding, UPDATED after the AUDIT-3 temporal-order fix
(fv_solver.py's step() now projects the predictor stage too --
incremental pressure correction, fixing a genuine 1st-order-in-time
splitting error).

ORIGINAL finding (still true as a general principle, kept below):
measuring TT-rank at a truncation tolerance TIGHTER than the solver's
own discretization error risks inflating the measured chi with noise,
not physics.

WHAT CHANGED: at Re=100, N=64, T=10, with the OLD (1st-order-in-time,
unprojected-predictor) solver, chi at tol=1e-8 was 21 -- inflated. With
the FIXED solver, chi at tol=1e-8 (and even tol=1e-12, and even at a
deliberately under-resolved N=32) is 5, matching D3's bound, with NO
inflation at any tested tolerance. The unprojected predictor was
injecting broadband, noise-like structure into the solution (spurious
non-physical divergence propagating through the corrector's RHS,
accumulated over many steps) -- exactly the kind of high-rank content a
loose tolerance hides and a tight one reveals. Fixing the temporal
splitting removed the noise SOURCE, not just its symptom.

Consequence: the general principle (tie rank-measurement tolerance to
the solver's own error, as a matter of methodology) remains correct
and is kept enforced in `scripts/measure_chi_of_re.py` -- but it is no
longer something this specific configuration can DEMONSTRATE, because
the fixed solver does not reproduce the failure mode. This test now
locks in the (better) new behavior instead of asserting a symptom that
no longer occurs.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from tgv.config import TGVParams, N_of_Re
from tgv.fv_solver import run
from tgv.quantics import quantics_reshape_2d, tt_ranks
from tgv.analytic import velocity
from tgv.norms import relative_l2_velocity


def test_chi_no_longer_inflated_at_tight_tolerance_after_temporal_order_fix():
    """Locks in the improved behavior: with the predictor-projection
    fix, chi=5 (matching D3) at every tested tolerance from 1e-12 to
    1e-2, including tolerances far tighter than the solver's own
    error -- the noise source that used to inflate chi is gone."""
    Re = 100
    p = TGVParams(Re=Re)
    N = N_of_Re(Re)
    dx = p.domain_length / N
    xs = (np.arange(N) + 0.5) * dx
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    u0, v0 = velocity(p, X, Y, 0.0)

    T = 10.0
    cfl = 0.3
    dt = min(cfl * dx / abs(p.Uc), cfl * dx**2 / (4 * p.nu))
    n_steps = max(1, int(np.ceil(T / dt)))
    dt = T / n_steps

    u, v, press, _ = run(u0, v0, dx, dx, p.nu, dt, n_steps=n_steps)
    u_ex, v_ex = velocity(p, X, Y, T)
    err = relative_l2_velocity(u, v, u_ex, v_ex)
    print(f"solver error at Re=100, N=64, T=10: {err:.4e}", file=sys.stderr)

    q_u = quantics_reshape_2d(u)
    for tol in (1e-12, 1e-8, 1e-4, 1e-2):
        chi = max(tt_ranks(q_u, tol=tol))
        print(f"tol={tol}: chi={chi}", file=sys.stderr)
        assert chi <= 6, (tol, chi)  # matches D3's chi<=5 bound (+/-1 slack), no inflation


def test_measure_chi_of_re_still_uses_error_matched_tolerance_as_a_principle():
    """The general methodological principle (rank tolerance should be
    tied to the solver's own error, never assumed to be safe at an
    arbitrary tight value) is still the right engineering practice even
    though this specific configuration no longer demonstrates a failure
    -- a future solver change could reintroduce noise, and the
    production script (scripts/measure_chi_of_re.py) still enforces the
    error-matched convention defensively."""
    import inspect
    from scripts import measure_chi_of_re

    source = inspect.getsource(measure_chi_of_re.measure_one)
    assert "rel_err" in source and "rank_tol" in source
