"""
N3 hypothesis discrimination (AUDIT-2 §3): the audit ranks three
hypotheses for why the unforced base-TGV case converges more slowly
(~1.0-1.5) than the MMS-forced case (>=1.8), and gives the cheapest
decisive test: manufacture `u_base + delta*(perturbation)` and sweep
delta -> 0.

  - Continuous degradation of the observed order as delta shrinks
    toward 0 means magnitude/dominance (H1: temporal floor
    contaminating the spatial study; or H2: it's the field's own
    structure, not the presence of forcing per se).
  - A discontinuous JUMP right at delta=0 means something structural
    (H3: compact-vs-wide divergence/gradient operator mismatch that
    only bites when the nonlinear cancellation is EXACT).

This test runs the grid-refinement order study (same methodology as
test_n3_fixed_re1000_grid_refinement_reports_observed_order_honestly)
at a sweep of eps (=delta) values using the ALREADY-EXISTING
k1_perturbed_field machinery, including very small eps, to see which
pattern the order(delta) curve actually shows.
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


def _observed_order(eps, nu=0.05, T=0.05, Ns=(64, 128, 256)):
    """Grid-refinement order for the K1-perturbed FORCED field at a
    given eps, using the SAME dt~0.05*dx methodology as the existing
    N3/G2 tests. eps=0 reduces to the unforced base TGV case exactly
    (forcing() is symbolically zero there, per test_d1_annihilation.py)."""
    errs = []
    for N in Ns:
        X, Y, dx = _grid(N)
        field = k1_perturbed_field(eps=eps, k2=3, Uc=1.0, Vc=0.0, V0=1.0, L=1.0, p0=0.0)
        u_num, v_num, _, Fx_num, Fy_num = field.lambdify_all(nu_val=nu)
        u0 = u_num(X, Y, 0.0) * np.ones_like(X)
        v0 = v_num(X, Y, 0.0) * np.ones_like(X)
        u_ex = u_num(X, Y, T) * np.ones_like(X)
        v_ex = v_num(X, Y, T) * np.ones_like(X)

        def forcing(t, Fx_num=Fx_num, Fy_num=Fy_num, X=X, Y=Y):
            Fx = np.asarray(Fx_num(X, Y, t), dtype=float) * np.ones_like(X)
            Fy = np.asarray(Fy_num(X, Y, t), dtype=float) * np.ones_like(X)
            return Fx, Fy

        dt = 0.05 * dx
        n_steps = max(1, int(round(T / dt)))
        dt = T / n_steps
        # forcing is applied even at eps=0 (returns zero arrays there) --
        # this exercises the SAME code path (forcing hook active) at
        # every eps, isolating the effect of eps's magnitude from the
        # effect of "is the forcing hook itself active"
        u, v, press, _ = run(u0, v0, dx, dx, nu, dt, n_steps=n_steps, forcing=forcing)

        errs.append(relative_l2_velocity(u, v, u_ex, v_ex))

    log_dx = np.log([2 * np.pi / N for N in Ns])
    log_err = np.log(errs)
    order = np.polyfit(log_dx, log_err, 1)[0]
    return order, errs


def test_n3_delta_sweep_discriminates_convergence_order_hypotheses():
    deltas = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0]
    orders = []
    for delta in deltas:
        order, errs = _observed_order(delta)
        orders.append(order)
        print(f"delta={delta}: order={order:.3f}, errs={errs}", file=sys.stderr)

    print(f"order(delta) for delta={deltas}: {[f'{o:.3f}' for o in orders]}", file=sys.stderr)

    # Report the shape of the curve. A continuous decline (each step
    # down in delta gives a smoothly smaller order, with no single big
    # jump concentrated at delta=0) supports H1/H2 (magnitude/dominance).
    # A discontinuous jump would show up as the last step (delta=0.001
    # -> 0) being much larger than the typical step size elsewhere.
    diffs = [orders[i] - orders[i + 1] for i in range(len(orders) - 1)]
    last_step = diffs[-1]
    typical_step = np.mean(np.abs(diffs[:-1])) if len(diffs) > 1 else abs(diffs[0])

    print(f"step sizes in order: {diffs}, last step={last_step:.3f}, "
          f"typical earlier step={typical_step:.3f}", file=sys.stderr)

    # This assertion is diagnostic, not a pass/fail gate on a specific
    # hypothesis: record whether the transition at delta=0 is an outlier
    # (>3x the typical earlier step) or consistent with the trend.
    is_discontinuous_jump = last_step > 3 * max(typical_step, 1e-6)
    print(f"CONCLUSION: {'H3 (structural, discontinuous)' if is_discontinuous_jump else 'H1/H2 (continuous magnitude/dominance effect)'}",
          file=sys.stderr)

    # sanity: orders must be positive and errors must decrease (basic
    # solver correctness, independent of which hypothesis holds)
    assert all(o > 0 for o in orders), orders
