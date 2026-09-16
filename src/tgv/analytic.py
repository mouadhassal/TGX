"""
Analytic (exact) solution of the convecting 2D Taylor-Green Vortex, and its
closed-form cell averages over axis-aligned rectangular control volumes.

All formulas are [FACT: airbus-tgv-constitution-v2.md] under the Q1 length
convention in config.py (L = 1, not the domain length).
"""
import numpy as np

from .config import TGVParams


def decay(p: TGVParams, t):
    """f(t) = exp(-2*nu*t/L^2)"""
    t = np.asarray(t, dtype=float)
    return np.exp(-2.0 * p.nu * t / p.L**2)


def velocity(p: TGVParams, x, y, t):
    """Exact pointwise velocity (u, v) at (x, y, t)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    f = decay(p, t)
    xt = (x - p.Uc * t) / p.L
    yt = (y - p.Vc * t) / p.L
    u = p.Uc + p.V0 * np.sin(xt) * np.cos(yt) * f
    v = p.Vc - p.V0 * np.cos(xt) * np.sin(yt) * f
    return u, v


def pressure(p: TGVParams, x, y, t):
    """Exact pointwise pressure. p* = p0 + (rho*V0^2/4)*f^2*(cos(2xt)+cos(2yt))"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    f = decay(p, t)
    xt = (x - p.Uc * t) / p.L
    yt = (y - p.Vc * t) / p.L
    return p.p0 + (p.rho * p.V0**2 / 4.0) * f**2 * (np.cos(2 * xt) + np.cos(2 * yt))


def mean_kinetic_energy(p: TGVParams, t):
    """E(t) = 1/2*(Uc^2+Vc^2) + (V0^2/4)*exp(-4*nu*t/L^2)

    Spatial mean of (u^2+v^2)/2 over one periodic cell; the fluctuating
    cross terms integrate to zero by periodicity/orthogonality.
    """
    t = np.asarray(t, dtype=float)
    f2 = np.exp(-4.0 * p.nu * t / p.L**2)
    return 0.5 * (p.Uc**2 + p.Vc**2) + (p.V0**2 / 4.0) * f2


def _sinc_avg_sin(a, b):
    """(1/(b-a)) * integral_a^b sin(s) ds = (cos(a) - cos(b)) / (b - a)"""
    return (np.cos(a) - np.cos(b)) / (b - a)


def _sinc_avg_cos(a, b):
    """(1/(b-a)) * integral_a^b cos(s) ds = (sin(b) - sin(a)) / (b - a)"""
    return (np.sin(b) - np.sin(a)) / (b - a)


def cell_average_velocity(p: TGVParams, x_lo, x_hi, y_lo, y_hi, t):
    """
    Exact cell average of (u, v) over the rectangle [x_lo,x_hi] x [y_lo,y_hi],
    in closed form (no quadrature): the mean-flow part is constant, and the
    oscillatory part factorizes into independent 1D sin/cos averages.

    WS0 step 2: this is the FV ground truth. Using point samples instead of
    this closed form silently injects an O(dx^2) error floor.
    """
    x_lo = np.asarray(x_lo, dtype=float)
    x_hi = np.asarray(x_hi, dtype=float)
    y_lo = np.asarray(y_lo, dtype=float)
    y_hi = np.asarray(y_hi, dtype=float)
    f = decay(p, t)

    ax_lo = (x_lo - p.Uc * t) / p.L
    ax_hi = (x_hi - p.Uc * t) / p.L
    ay_lo = (y_lo - p.Vc * t) / p.L
    ay_hi = (y_hi - p.Vc * t) / p.L

    sin_x_avg = _sinc_avg_sin(ax_lo, ax_hi)
    cos_x_avg = _sinc_avg_cos(ax_lo, ax_hi)
    sin_y_avg = _sinc_avg_sin(ay_lo, ay_hi)
    cos_y_avg = _sinc_avg_cos(ay_lo, ay_hi)

    u_avg = p.Uc + p.V0 * sin_x_avg * cos_y_avg * f
    v_avg = p.Vc - p.V0 * cos_x_avg * sin_y_avg * f
    return u_avg, v_avg
