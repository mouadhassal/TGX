"""
G0 (constitution v2 §11): reproduce the organizers' two supplied figures
under the Q1 convention BEFORE anything downstream is trusted.

  - KE(t=0)  = 0.75  for any Re
  - KE(t=10) ~= 0.52 at Re = 100
  - pressure extremum at t=1, Re=100 matches a +/-0.4 colourbar
    (0.5 * exp(-4*nu*t) ~= 0.39)

If this file fails, Q1 (nu = 2*pi/Re, L = 1) is wrong and every number
downstream in the plan is invalid -- stop and escalate to the organizers.
"""
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tgv.config import TGVParams
from tgv.analytic import mean_kinetic_energy, pressure, cell_average_velocity, velocity


def test_g0_ke_at_t0_is_075_for_any_re():
    for Re in (10, 100, 1000):
        p = TGVParams(Re=Re)
        E0 = mean_kinetic_energy(p, 0.0)
        assert math.isclose(E0, 0.75, rel_tol=1e-9), (Re, E0)


def test_g0_ke_at_t10_re100_matches_figure():
    p = TGVParams(Re=100)
    E10 = mean_kinetic_energy(p, 10.0)
    assert math.isclose(E10, 0.52, rel_tol=0.02), E10  # figure read to ~2 sig figs


def test_g0_pressure_extremum_at_t1_re100_matches_colourbar():
    p = TGVParams(Re=100)
    # extremum of p* over space is +/- rho*V0^2/4 * f(t)^2 * 2 = 0.5*f^2 at t=1
    f1 = math.exp(-2.0 * p.nu * 1.0 / p.L**2)
    extremum = 0.5 * f1**2
    assert math.isclose(extremum, 0.39, rel_tol=0.05), extremum
    # cross-check against the actual pressure field, at the point where
    # xt = (x - Uc*t)/L = 0 and yt = (y - Vc*t)/L = 0 so cos(2xt)+cos(2yt) = 2
    x, y = p.Uc * 1.0, p.Vc * 1.0
    p_at_extremum = pressure(p, x, y, 1.0)
    assert math.isclose(p_at_extremum, extremum, rel_tol=1e-9)


def test_cell_average_reduces_to_point_value_as_cell_shrinks():
    p = TGVParams(Re=100)
    x0, y0, t = 1.3, 2.1, 0.7
    eps = 1e-5
    u_avg, v_avg = cell_average_velocity(p, x0 - eps, x0 + eps, y0 - eps, y0 + eps, t)
    u_pt, v_pt = velocity(p, x0, y0, t)
    assert math.isclose(u_avg, u_pt, abs_tol=1e-4)
    assert math.isclose(v_avg, v_pt, abs_tol=1e-4)


def test_cell_average_matches_numerical_quadrature():
    from scipy import integrate

    p = TGVParams(Re=100)
    x_lo, x_hi, y_lo, y_hi, t = 0.2, 1.1, 0.5, 1.9, 2.0

    def u_fn(y, x):
        u, _ = velocity(p, x, y, t)
        return u

    val, _ = integrate.dblquad(u_fn, x_lo, x_hi, y_lo, y_hi)
    expected_avg = val / ((x_hi - x_lo) * (y_hi - y_lo))

    u_avg, _ = cell_average_velocity(p, x_lo, x_hi, y_lo, y_hi, t)
    assert math.isclose(u_avg, expected_avg, rel_tol=1e-8)
