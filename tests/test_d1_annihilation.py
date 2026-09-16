"""
D1 (constitution v2 §2, "nonlinearity is annihilated"): independent
symbolic re-derivation, required by Rule 2 before D1 may appear in any
draft. Also feeds G1 (>=4 of 6 degeneracies survive symbolic derivation).
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import sympy as sp
from tgv.mms import base_tgv_field, k1_perturbed_field, x, y, t


def test_d1_base_tgv_is_exact_unforced_solution():
    """The challenge's exact solution must solve the UNFORCED
    incompressible NS equations identically -- forcing == 0 -- which is
    D1's precise claim (nonlinear term is exactly a pressure gradient)."""
    field = base_tgv_field()
    Fx, Fy = field.forcing()
    assert Fx == 0, Fx
    assert Fy == 0, Fy
    assert field.divergence() == 0


def test_d1_convective_term_equals_pressure_gradient():
    """Direct symbolic check of the theorem statement: (u*.grad)u* minus
    the mean-flow advection term equals -grad(p*)/rho exactly."""
    field = base_tgv_field()
    u, v, p = field.u, field.v, field.p

    conv_u = u * sp.diff(u, x) + v * sp.diff(u, y)
    conv_v = u * sp.diff(v, x) + v * sp.diff(v, y)

    Uc, Vc = 1, 0
    fluct_u = sp.simplify(conv_u - Uc * sp.diff(u, x) - Vc * sp.diff(u, y))
    fluct_v = sp.simplify(conv_v - Uc * sp.diff(v, x) - Vc * sp.diff(v, y))

    rho = sp.Symbol("rho", real=True)
    dpdx = sp.diff(p, x)
    dpdy = sp.diff(p, y)

    assert sp.simplify(fluct_u + dpdx / rho) == 0
    assert sp.simplify(fluct_v + dpdy / rho) == 0


def test_d1_k1_perturbed_field_breaks_the_cancellation():
    """K1 knob sanity check: at eps=0 the perturbed field reduces to the
    exact unforced solution; at eps!=0 the forcing is generically
    nonzero, confirming K1 actually breaks D1 rather than being a no-op."""
    eps = sp.Symbol("epsilon", real=True)
    field = k1_perturbed_field(eps=eps, k2=3)

    Fx0, Fy0 = field.forcing()
    Fx_at_zero = sp.simplify(Fx0.subs(eps, 0))
    Fy_at_zero = sp.simplify(Fy0.subs(eps, 0))
    assert Fx_at_zero == 0
    assert Fy_at_zero == 0

    # forcing must depend on eps (not identically zero for eps != 0)
    assert Fx0.has(eps) or Fy0.has(eps)
