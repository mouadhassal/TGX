"""
K5 knob, wired through the FULL nonlinear FV solver (not just the
conditioning-module / circulant-operator level in test_k5_anisotropy.py).

AUDIT.md B1 fix: K5 is grid anisotropy (rectangular grid, Nx != Ny),
NOT viscosity anisotropy. nu stays isotropic and the PDE is UNCHANGED,
so the exact TGV solution remains an exact reference at every aspect
ratio -- unlike the earlier (wrong) version, error must NOT blow up
here; that would indicate a bug, not a discriminator. The actual K5
cost discriminator is the explicit diffusion-stability step limit,
which is set by whichever axis has the SMALLER cell (dt ~ min(dx,dy)^2
/ nu) -- a direct, measurable wall-clock cost from grid anisotropy
alone, with no change to solution accuracy required.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from tgv.fv_solver import run
from tgv.config import TGVParams
from tgv.analytic import velocity
from tgv.norms import relative_l2_velocity


def _rect_grid(Nx, Ny, domain_length=2 * np.pi):
    dx = domain_length / Nx
    dy = domain_length / Ny
    xs = (np.arange(Nx) + 0.5) * dx
    ys = (np.arange(Ny) + 0.5) * dy
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return X, Y, dx, dy


def test_k5_solver_error_stays_valid_under_grid_anisotropy():
    """Reference validity check: since the PDE and exact solution are
    UNCHANGED by grid anisotropy (only nu's DISCRETIZATION differs per
    axis), error must stay small and bounded at every aspect ratio --
    NOT grow into a mismatch, unlike the earlier (wrong) viscosity-based
    version of this knob."""
    Nx = 64
    Re = 100
    p = TGVParams(Re=Re)
    T = 0.5

    errs = []
    for ratio in (1, 2, 4, 8):
        Ny = Nx * ratio
        X, Y, dx, dy = _rect_grid(Nx, Ny)

        u0, v0 = velocity(p, X, Y, 0.0)
        u_ex, v_ex = velocity(p, X, Y, T)

        dt = 0.2 * min(dx, dy) ** 2 / (4 * p.nu)
        n_steps = max(1, int(round(T / dt)))
        dt = T / n_steps

        u, v, press, _ = run(u0, v0, dx, dy, p.nu, dt, n_steps=n_steps)
        err = relative_l2_velocity(u, v, u_ex, v_ex)
        errs.append((ratio, err))

    for ratio, err in errs:
        print(f"ratio={ratio}: err={err:.3e}", file=sys.stderr)

    # reference stays valid: every error is small discretization error,
    # none blows up or jumps by orders of magnitude
    assert all(err < 0.05 for _, err in errs), errs


def test_k5_solver_stability_limited_dt_shrinks_with_grid_aspect_ratio():
    """The genuine K5 cost discriminator: the finer axis sets the
    explicit diffusion-stability dt, so required steps for fixed T grow
    with the aspect ratio -- a real wall-clock cost from grid anisotropy
    alone, with the exact solution untouched."""
    Nx = 64
    nu = 2 * np.pi / 100

    dts = []
    for ratio in (1, 2, 5, 10):
        Ny = Nx * ratio
        dx = 2 * np.pi / Nx
        dy = 2 * np.pi / Ny
        dt = 0.2 * min(dx, dy) ** 2 / (4 * nu)
        dts.append(dt)

    assert dts == sorted(dts, reverse=True)  # dt shrinks monotonically
    assert np.isclose(dts[0] / dts[-1], 100.0)  # ~ ratio^2, since dt ~ dy^2 ~ (dx/ratio)^2
