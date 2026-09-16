"""
K2 knob, wired through the FULL nonlinear FV solver (not just the
circulant fast-forward operator in test_k2_breaks_fastforward.py).

Uses k2_time_dependent_convection_field (mms.py): Uc(t) = Uc0 +
amplitude*sin(omega*t), with the traveling-wave phase correctly tracking
the actual displacement integral. At amplitude=0 this is exactly the
challenge's own field (D1/D2 corner); its forcing is symbolically zero
(confirmed in mms.py's docstring derivation). We run the UNFORCED
solver -- which only ever sees an exact solution at amplitude=0 -- and
show its error against the true (still time-varying, for amplitude>0)
exact reference grows once amplitude departs from zero.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from tgv.fv_solver import run
from tgv.mms import k2_time_dependent_convection_field
from tgv.norms import relative_l2_velocity


def _grid(N, domain_length=2 * np.pi):
    dx = domain_length / N
    xs = (np.arange(N) + 0.5) * dx
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    return X, Y, dx


def test_k2_solver_error_flat_at_amplitude_zero_then_grows():
    N = 64
    X, Y, dx = _grid(N)
    nu = 0.05
    T = 0.5

    errs = []
    for amplitude in (0.0, 0.2, 0.5, 1.0):
        field = k2_time_dependent_convection_field(Uc0=1.0, amplitude=amplitude, omega=2,
                                                     Vc=0.0, V0=1.0, L=1.0, p0=0.0)
        u_num, v_num, _, _, _ = field.lambdify_all(nu_val=nu)
        u0 = u_num(X, Y, 0.0) * np.ones_like(X)
        v0 = v_num(X, Y, 0.0) * np.ones_like(X)

        dt = 0.2 * dx**2 / (4 * nu)
        n_steps = max(1, int(round(T / dt)))
        dt = T / n_steps

        # UNFORCED: never sees K2's forcing term (which is nonzero for
        # amplitude != 0, per mms.py's symbolic derivation)
        u, v, press, _ = run(u0, v0, dx, dx, nu, dt, n_steps=n_steps)

        u_ex = u_num(X, Y, T) * np.ones_like(X)
        v_ex = v_num(X, Y, T) * np.ones_like(X)
        err = relative_l2_velocity(u, v, u_ex, v_ex)
        errs.append((amplitude, err))

    for amplitude, err in errs:
        print(f"amplitude={amplitude}: err={err:.3e}", file=sys.stderr)

    err0 = errs[0][1]
    err_max = errs[-1][1]

    assert err0 < 5e-3, errs           # flat: pure discretization floor (D1/D2 corner)
    assert err_max > 20 * err0, errs   # jump: grows substantially once amplitude > 0
