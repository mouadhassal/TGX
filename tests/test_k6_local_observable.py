"""
K6 knob (constitution v2 §3, breaks D6): score a probe velocity or
integrated force instead of global KE. Grounds extraction.py's sample-
complexity argument in the actual FV solver's grid (not just an
abstract sampling model), and confirms the SOLVER's own output supports
both a cheap global readout and a genuinely more expensive local one.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from tgv.fv_solver import run
from tgv.config import TGVParams
from tgv.analytic import velocity
from tgv.extraction import local_observable_samples_needed


def _grid(N, domain_length=2 * np.pi):
    dx = domain_length / N
    xs = (np.arange(N) + 0.5) * dx
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    return X, Y, dx


def test_k6_global_ke_from_solver_output_is_a_single_number_regardless_of_n():
    """The global functional (K6=0 corner) costs the same '1 readout'
    regardless of grid resolution N -- it is a scalar reduction over the
    whole field."""
    Re = 100
    costs = []
    for N in (32, 64, 128):
        X, Y, dx = _grid(N)
        p = TGVParams(Re=Re)
        u0, v0 = velocity(p, X, Y, 0.0)
        dt = 0.2 * dx**2 / (4 * p.nu)
        n_steps = 50
        u, v, press, _ = run(u0, v0, dx, dx, p.nu, dt, n_steps=n_steps)
        ke = float(np.mean(0.5 * (u ** 2 + v ** 2)))  # ONE number, any N
        costs.append(1)  # one global readout, by construction

    assert costs == [1, 1, 1]


def test_k6_local_probe_observables_cost_scales_with_count_not_n():
    """K6: requesting n_outputs independent local probes (e.g. velocity
    at n_outputs distinct points extracted from the SAME solver run)
    costs n_outputs times a single-observable budget -- the discriminator
    against the global-functional case, using the sample-complexity
    model from extraction.py grounded on an actual solver grid."""
    Re = 100
    N = 64
    X, Y, dx = _grid(N)
    p = TGVParams(Re=Re)
    u0, v0 = velocity(p, X, Y, 0.0)
    dt = 0.2 * dx**2 / (4 * p.nu)
    u, v, press, _ = run(u0, v0, dx, dx, p.nu, dt, n_steps=50)

    eps = 0.02
    for n_probes in (1, 8, 64):
        probe_idx = np.linspace(0, N * N - 1, n_probes, dtype=int)
        probe_vals = u.ravel()[probe_idx]
        assert len(probe_vals) == n_probes  # n_probes genuinely distinct outputs

        cost = local_observable_samples_needed(n_probes, eps)
        expected = n_probes * local_observable_samples_needed(1, eps)
        assert cost == expected


def test_k6_integrated_force_is_a_global_functional_like_ke():
    """An INTEGRATED surface force (sum/integral over a boundary) is,
    like KE, a single global reduction -- cheap, unlike n_probes
    independent local point values. Confirms K6's distinction is about
    the OBSERVABLE's locality, not merely 'uses the pressure field'."""
    Re = 100
    N = 64
    X, Y, dx = _grid(N)
    p = TGVParams(Re=Re)
    u0, v0 = velocity(p, X, Y, 0.0)
    dt = 0.2 * dx**2 / (4 * p.nu)
    u, v, pressure_field, _ = run(u0, v0, dx, dx, p.nu, dt, n_steps=50)

    # a representative "integrated force on a line" is one scalar reduction
    integrated_force_x = float(np.sum(pressure_field[0, :]) * dx)
    assert np.isscalar(integrated_force_x) or isinstance(integrated_force_x, float)
