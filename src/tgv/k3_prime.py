"""
K3' (AUDIT.md S1): reinstates a knob that stresses D3 (the untested
degeneracy underneath the §7.1 headline) after K3 (prescribed spectral
slope, MMS-forced) was cut by its own kill criterion.

Mechanism: on the LINEAR advection-diffusion path (where D2 and D3
themselves live -- this is the equation D1 reduces the full NS momentum
equation TO, not the full nonlinear NS equations), superposition is
exact: each mode
    a_j * sin(k_j*(x-Uc*t)) * cos(k_j*(y-Vc*t)) * exp(-2*nu*k_j^2*t)
individually solves the linear advection-diffusion equation for ANY
k_j, so a sum of m such modes does too, by linearity -- ZERO
manufactured forcing required, unlike K1/K2/K5's nonlinear-equation
knobs. Rank grows with m, tunably, with an exact reference at every m.
"""
import numpy as np
import sympy as sp


def occupied_subspace_kappa_multimode(ks) -> float:
    """
    AUDIT-2 §4: K5 (grid anisotropy) CANNOT break D5 -- the TGV occupies
    ONE mode, so the operator restricted to the occupied subspace is a
    1x1 block, and the condition number of a 1x1 block is exactly 1 for
    every grid, every aspect ratio, every Re. No grid deformation can
    move it; K5's occupied-subspace kappa staying pinned at ~1 was never
    evidence of anything (see conditioning.py, K5 docstrings, now
    corrected to say so).

    K3' genuinely CAN break D5: with m incommensurate modes k_1..k_m,
    the occupied subspace is m-dimensional with diffusive decay rates
    nu*k_j^2 (real part of the dispersion relation, same convention as
    conditioning.py's D5 functions -- advection is excluded, being
    purely imaginary/skew and irrelevant to conditioning). Its condition
    number is
        kappa = max_j(k_j^2) / min_j(k_j^2)
    which is 1 at m=1 (matching D5's single-mode claim exactly) and
    grows by construction as incommensurate modes are added -- directly
    tunable, unlike K5.
    """
    ks = np.asarray(ks, dtype=float)
    k2 = ks ** 2
    return float(np.max(k2) / np.min(k2))


def multi_mode_field(X, Y, t, ks, amplitudes, Uc=1.0, Vc=0.0, nu=0.05, phases=None):
    """Numpy evaluation of the m-mode exact linear-advection-diffusion
    solution at grid points (X, Y) and time t."""
    if phases is None:
        phases = np.zeros(len(ks))
    field = np.zeros_like(X, dtype=float)
    xt = X - Uc * t
    yt = Y - Vc * t
    for k, a, phase in zip(ks, amplitudes, phases):
        field += a * np.sin(k * xt + phase) * np.cos(k * yt) * np.exp(-2 * nu * k ** 2 * t)
    return field


def verify_exact_symbolically(ks, amplitudes, Uc=1.0, Vc=0.0, nu_val=0.05, phases=None):
    """
    Symbolic proof (not merely a numerical spot-check) that the
    m-mode superposition exactly solves the LINEAR advection-diffusion
    equation d/dt(u) + Uc*du/dx + Vc*du/dy - nu*lap(u) = 0, for the
    given mode set. Returns the simplified residual (must be exactly 0
    sympy.Integer(0) for the claim to hold).
    """
    x, y, t, nu = sp.symbols("x y t nu", real=True)
    if phases is None:
        phases = [0] * len(ks)

    # exact rational arithmetic throughout -- sp.Float here would leave
    # ~1e-17 floating-point noise after simplify() instead of an exact
    # symbolic 0, which is not what Rule 2's "independently re-derive"
    # standard asks for.
    Uc_r, Vc_r, nu_val_r = sp.nsimplify(Uc), sp.nsimplify(Vc), sp.nsimplify(nu_val)

    xt = x - Uc_r * t
    yt = y - Vc_r * t
    u = sum(
        sp.nsimplify(a) * sp.sin(sp.Integer(k) * xt + sp.nsimplify(ph)) * sp.cos(sp.Integer(k) * yt)
        * sp.exp(-2 * nu * sp.Integer(k) ** 2 * t)
        for k, a, ph in zip(ks, amplitudes, phases)
    )

    residual = sp.diff(u, t) + Uc_r * sp.diff(u, x) + Vc_r * sp.diff(u, y) - nu * (
        sp.diff(u, x, 2) + sp.diff(u, y, 2)
    )
    residual = sp.simplify(residual.subs(nu, nu_val_r))
    return residual
