"""
WS2 / §7.1: measure chi(Re) from the ACTUAL dense FV solver's output
(not just the exact analytic field, already checked in
tests/test_d3_quantics_rank.py), across the pre-registered Re grid
(protocol.md), at feasible resolution. Then extrapolate the memory
comparison to Re = 10^6 exactly as the constitution's §7.1 derivation
does, so that derivation is grounded in code that has actually run
rather than only algebra.

Per Rule 3: this uses the pre-registered N(Re) rule and T; per Rule 6,
records machine spec. Per Rule 1, every number below is tagged.
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from tgv.config import TGVParams, N_of_Re
from tgv.fv_solver import run
from tgv.quantics import quantics_reshape_2d, tt_ranks
from tgv.timing import MachineSpec


def measure_one(Re: float, T: float = 10.0, cfl: float = 0.3):
    p = TGVParams(Re=Re)
    N = N_of_Re(Re)
    dx = p.domain_length / N
    xs = (np.arange(N) + 0.5) * dx
    X, Y = np.meshgrid(xs, xs, indexing="ij")

    from tgv.analytic import velocity
    u0, v0 = velocity(p, X, Y, 0.0)

    dt_adv = cfl * dx / abs(p.Uc)
    dt_diff = cfl * dx**2 / (4 * p.nu)
    dt = min(dt_adv, dt_diff)
    n_steps = max(1, int(np.ceil(T / dt)))
    dt = T / n_steps

    t0 = time.perf_counter()
    u, v, press, diags = run(u0, v0, dx, dx, p.nu, dt, n_steps=n_steps)
    wall_s = time.perf_counter() - t0

    max_div = max(d.max_div for d in diags)

    # rank-truncation tolerance MUST be tied to this run's own
    # discretization error, never an arbitrary tight number -- see
    # tests/test_chi_measurement_tolerance.py. A tolerance far below the
    # solver's own L2 error measures truncation-error noise, not the
    # field's physical rank (chi=21 vs chi=5 at Re=100 in that test).
    from tgv.norms import relative_l2_velocity
    u_ex, v_ex = velocity(p, X, Y, T)
    rel_err = relative_l2_velocity(u, v, u_ex, v_ex)
    rank_tol = max(rel_err, 1e-8)

    q_u = quantics_reshape_2d(u)
    ranks_u = tt_ranks(q_u, tol=rank_tol)
    chi = int(max(ranks_u)) if ranks_u else 1

    n_bits = int(round(np.log2(N)))
    dense_bytes = 2 * N * N * 8  # u and v, float64
    # quantics storage: 2 fields, 2n sites (x and y bits interleaved -> 2n
    # total quantics indices for the 2D field), each core <= chi*chi*2*8B
    quantics_bytes = 2 * (2 * n_bits) * (chi ** 2) * 8

    return {
        "Re": float(Re),
        "N": int(N),
        "n_steps": int(n_steps),
        "T": float(T),
        "wall_s": float(wall_s),
        "max_div": float(max_div),
        "rel_l2_error": float(rel_err),
        "rank_tol_used": float(rank_tol),
        "chi": chi,
        "dense_bytes": int(dense_bytes),
        "quantics_bytes": int(quantics_bytes),
    }


def extrapolate_re1e6():
    """[DERIVED, §7.1]: N(1e6) = 2^ceil(log2(max(32,5e5))) grid, dense
    vs quantics storage assuming chi stays <= 5 (measured to hold at all
    feasible Re above; NOT independently run at Re=1e6 -- infeasible)."""
    Re = 1e6
    N = N_of_Re(Re)
    n_bits = int(round(np.log2(N)))
    chi_assumed = 5  # per D3, measured constant across feasible Re
    dense_bytes = 2 * N * N * 8
    quantics_bytes = 2 * (2 * n_bits) * (chi_assumed ** 2) * 8
    return {
        "Re": Re, "N": N, "n_bits": n_bits, "chi_assumed": chi_assumed,
        "dense_bytes": dense_bytes, "quantics_bytes": quantics_bytes,
    }


if __name__ == "__main__":
    results = []
    for Re in (10, 100, 1000):
        print(f"[running] Re={Re} ...", file=sys.stderr)
        r = measure_one(Re)
        results.append(r)
        print(json.dumps(r, indent=2))

    extrap = extrapolate_re1e6()

    out = {
        "machine": MachineSpec().__dict__,
        "measured": results,
        "extrapolated_Re_1e6": extrap,
    }

    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "chi_of_re.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}", file=sys.stderr)
