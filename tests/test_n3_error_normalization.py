"""
N3 (AUDIT.md): the required "error scaling vs Re" curve, as originally
normalized (relative_l2_velocity, dividing by the TOTAL field's norm
including the non-decaying mean flow), conflates discretization quality
with how much of the vortex survives to T=10 at each Re -- at low Re
the vortex has decayed to a few percent of its initial amplitude by
T=10, so the SAME absolute error reads as a much LARGER relative error
purely from a smaller denominator, not worse numerics. This is why
chi_of_re.json shows error INCREASING with N (1.9e-7 -> 4.0e-3 -> 9.1e-3
at N=32,64,512): different amounts of surviving signal, same
normalization convention.

Fixes: (1) relative_l2_fluctuation (norms.py), normalizing by the
fluctuation's own norm; (2) a genuine FIXED-Re grid-refinement study,
RESOLVED per AUDIT-2 §3 (H1 confirmed: dt~dx was not aggressive enough
in suppressing temporal error for the unforced case; dt~dx^2 recovers
clean 2nd order at both Re=100 and Re=1000) -- see
test_n3_fixed_re1000_grid_refinement_confirms_second_order_with_dt_squared_scaling.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from tgv.fv_solver import run
from tgv.config import TGVParams
from tgv.analytic import velocity
from tgv.norms import relative_l2_velocity, relative_l2_fluctuation


def _grid(N, domain_length=2 * np.pi):
    dx = domain_length / N
    xs = (np.arange(N) + 0.5) * dx
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    return X, Y, dx


def test_n3_fluctuation_normalized_error_decreases_with_resolution_at_fixed_short_t():
    """Unlike the total-normalized error (which INCREASES with N in
    chi_of_re.json, a vortex-survival artifact), the fluctuation-
    normalized error must behave like genuine discretization error:
    decreasing (or flat) as resolution improves, at a FIXED, matched Re
    and short T (so vortex survival is comparable across N, isolating
    the normalization effect from Re-dependence)."""
    Re = 100
    p = TGVParams(Re=Re)
    T = 0.1

    total_errs, fluct_errs = [], []
    for N in (32, 64, 128):
        X, Y, dx = _grid(N)
        u0, v0 = velocity(p, X, Y, 0.0)
        dt = 0.2 * dx ** 2 / (4 * p.nu)
        n_steps = max(1, int(round(T / dt)))
        dt = T / n_steps
        u, v, press, _ = run(u0, v0, dx, dx, p.nu, dt, n_steps=n_steps)

        u_ex, v_ex = velocity(p, X, Y, T)
        total_errs.append(relative_l2_velocity(u, v, u_ex, v_ex))
        fluct_errs.append(relative_l2_fluctuation(u, v, u_ex, v_ex, p.Uc, p.Vc))

    print(f"total-normalized errors: {total_errs}", file=sys.stderr)
    print(f"fluctuation-normalized errors: {fluct_errs}", file=sys.stderr)

    assert fluct_errs == sorted(fluct_errs, reverse=True), fluct_errs  # decreases with N
    # fluctuation-normalized error must be LARGER than total-normalized
    # (since the fluctuation's own norm is smaller than the total
    # field's norm, being normalized against a smaller denominator)
    assert all(f >= t for f, t in zip(fluct_errs, total_errs)), (fluct_errs, total_errs)


def test_n3_fixed_re1000_grid_refinement_confirms_second_order_with_dt_squared_scaling():
    """
    RESOLVED (AUDIT-2 §3, H1 confirmed): the previously-open finding
    ("unforced base-TGV case converges at only ~1.0-1.5, mechanism not
    understood") is closed. AUDIT-2 ranked H1 ("the temporal error floor
    is contaminating the spatial study -- you identified exactly this
    artifact in N2 for naive dt-halving, then did not apply the same
    correction to the spatial refinement") as most likely, with a
    one-sweep decisive test: refine with dt ~ dx^2 instead of dt ~ dx.

    Result: with dt ~ dx^2 (0.1*dx^2, well under the diffusive stability
    limit), the unforced case's observed order is 1.994 at Re=100 and
    1.998 at Re=1000 -- genuine 2nd order, matching G2's MMS-forced
    result. The prior dt~dx (0.05*dx) scaling suppressed temporal error
    only linearly faster than the dx^2 spatial term shrinks, so at fine
    N the O(dt) component (dt~dx here) ends up comparable to or larger
    than the shrinking O(dx^2) spatial term, dragging the apparent
    order down toward 1 -- exactly H1's mechanism. The MMS-forced case
    happened to look clean under the same (insufficiently strict) dt~dx
    scaling because its larger spatial error (a richer field) stays
    above that floor longer over the SAME tested N range (AUDIT-2's own
    explanation, confirmed).

    The complementary delta-sweep (test_n3_delta_sweep.py) independently
    rules out H3 (a structural jump right at zero forcing): order(delta)
    declines smoothly from 1.75 (delta=0.3) to 1.06 (delta=0) with no
    discontinuity, consistent with a magnitude/dominance effect (H1/H2)
    rather than a structural one -- and H1 alone, confirmed here, fully
    explains it without invoking H2.
    """
    for Re in (100, 1000):
        p = TGVParams(Re=Re)
        T = 0.05
        Ns = (64, 128, 256)

        errs = []
        for N in Ns:
            X, Y, dx = _grid(N)
            u0, v0 = velocity(p, X, Y, 0.0)
            dt = 0.1 * dx ** 2  # dt ~ dx^2, not dt ~ dx -- the H1 fix
            n_steps = max(1, int(round(T / dt)))
            dt = T / n_steps
            u, v, press, _ = run(u0, v0, dx, dx, p.nu, dt, n_steps=n_steps)

            u_ex, v_ex = velocity(p, X, Y, T)
            errs.append(relative_l2_velocity(u, v, u_ex, v_ex))

        log_dx = np.log([2 * np.pi / N for N in Ns])
        log_err = np.log(errs)
        order = np.polyfit(log_dx, log_err, 1)[0]
        print(f"Re={Re}, dt~dx^2, N={Ns}: errors={errs}, order={order:.3f}", file=sys.stderr)

        assert errs == sorted(errs, reverse=True), (Re, errs)
        assert order >= 1.8, (Re, errs, order)  # now genuinely supported by evidence
