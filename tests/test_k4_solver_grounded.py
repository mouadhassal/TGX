"""
K4 knob, grounded in the actual FV solver (not just the analytic
formula already checked in test_d4_nonunitarity.py): running the solver
itself at Uc=1 vs Uc=0 must reproduce the bounded-vs-unbounded KE decay
behavior D4/K4 predict, confirming the solver's own discretization
doesn't accidentally hide or exaggerate the effect.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from tgv.fv_solver import run
from tgv.config import TGVParams
from tgv.analytic import velocity


def _grid(N, domain_length=2 * np.pi):
    dx = domain_length / N
    xs = (np.arange(N) + 0.5) * dx
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    return X, Y, dx


def _mean_ke(u, v):
    return float(np.mean(0.5 * (u ** 2 + v ** 2)))


def test_k4_solver_reproduces_bounded_vs_unbounded_ke_decay():
    N = 32
    X, Y, dx = _grid(N)
    Re = 100
    T = 5.0

    ke_ratios = {}
    for Uc in (1.0, 0.0):
        p = TGVParams(Re=Re, Uc=Uc, Vc=0.0)
        u0, v0 = velocity(p, X, Y, 0.0)
        ke0 = _mean_ke(u0, v0)

        dt = 0.2 * dx**2 / (4 * p.nu)  # diffusion-limited, safe for both cases
        n_steps = max(1, int(round(T / dt)))
        dt = T / n_steps

        u, v, press, _ = run(u0, v0, dx, dx, p.nu, dt, n_steps=n_steps)
        keT = _mean_ke(u, v)
        ke_ratios[Uc] = np.sqrt(keT / ke0)

    # Uc=1 (challenge parameters): bounded, norm ratio stays > sqrt(2/3)
    # (approached from above as T grows, per D4's floor derivation)
    assert ke_ratios[1.0] > np.sqrt(2.0 / 3.0) - 0.02, ke_ratios

    # Uc=0 (K4): unbounded decay -- at T=5 it must already be measurably
    # below the Uc=1 case, confirming the solver itself (not just the
    # closed-form formula) shows the restored non-unitarity difficulty
    assert ke_ratios[0.0] < ke_ratios[1.0] - 0.2, ke_ratios


def test_k4_solver_uc_zero_ratio_keeps_decaying_toward_zero():
    """Unlike Uc=1 (which the solver above shows approaching a bounded
    floor), Uc=0's norm ratio must continue falling as T grows -- no
    floor exists once the surviving mean flow is removed."""
    N = 32
    X, Y, dx = _grid(N)
    Re = 100

    ratios = []
    for T in (2.0, 5.0, 10.0):
        p = TGVParams(Re=Re, Uc=0.0, Vc=0.0)
        u0, v0 = velocity(p, X, Y, 0.0)
        ke0 = _mean_ke(u0, v0)

        dt = 0.2 * dx**2 / (4 * p.nu)
        n_steps = max(1, int(round(T / dt)))
        dt = T / n_steps

        u, v, press, _ = run(u0, v0, dx, dx, p.nu, dt, n_steps=n_steps)
        keT = _mean_ke(u, v)
        ratios.append(np.sqrt(keT / ke0))

    assert ratios == sorted(ratios, reverse=True), ratios
    assert ratios[-1] < 0.4 * ratios[0], ratios
