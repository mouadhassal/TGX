"""
Symbolic method-of-manufactured-solutions (MMS) forcing generator
(WS0.4): the engine that drives all six knobs K1-K6 and the WS1 proof
pack. Given any candidate velocity/pressure field (u, v, p) as sympy
expressions in (x, y, t; nu, rho), returns the exact forcing terms that
make it a solution of the forced incompressible Navier-Stokes equations:

    du/dt + (u.grad)u = -grad(p)/rho + nu*lap(u) + F
    div(u) = 0   (checked, not forced -- see `divergence`)

Setting F = 0 and checking it is exactly the D1 annihilation theorem
(constitution v2 D1): substitute the unforced TGV solution and verify
Fx = Fy = 0 identically.
"""
from dataclasses import dataclass

import sympy as sp

x, y, t, nu, rho = sp.symbols("x y t nu rho", real=True)
nu_x, nu_y = sp.symbols("nu_x nu_y", real=True)  # K5: anisotropic diffusion


@dataclass
class MMSField:
    u: sp.Expr
    v: sp.Expr
    p: sp.Expr

    def divergence(self) -> sp.Expr:
        return sp.simplify(sp.diff(self.u, x) + sp.diff(self.v, y))

    def forcing(self, nu_x_sym=None, nu_y_sym=None):
        """
        Return (Fx, Fy), fully simplified, such that (u, v, p) exactly
        solves the forced momentum equations for the given diffusion
        coefficients.

        nu_x_sym, nu_y_sym: K5 (anisotropic diffusion) generalization --
        default to the single isotropic `nu` symbol for BOTH directions
        (unchanged behavior for every existing caller/test); pass
        distinct symbols (e.g. module-level `nu_x`, `nu_y`) to get the
        anisotropic Laplacian nu_x*d2u/dx2 + nu_y*d2u/dy2 instead of
        nu*(d2u/dx2+d2u/dy2).
        """
        if nu_x_sym is None:
            nu_x_sym = nu
        if nu_y_sym is None:
            nu_y_sym = nu

        u, v, p = self.u, self.v, self.p

        conv_u = u * sp.diff(u, x) + v * sp.diff(u, y)
        conv_v = u * sp.diff(v, x) + v * sp.diff(v, y)

        lap_u = nu_x_sym * sp.diff(u, x, 2) + nu_y_sym * sp.diff(u, y, 2)
        lap_v = nu_x_sym * sp.diff(v, x, 2) + nu_y_sym * sp.diff(v, y, 2)

        Fx = sp.diff(u, t) + conv_u + sp.diff(p, x) / rho - lap_u
        Fy = sp.diff(v, t) + conv_v + sp.diff(p, y) / rho - lap_v

        return sp.simplify(Fx), sp.simplify(Fy)

    def is_exact_unforced_solution(self) -> bool:
        """True iff this field solves the UNFORCED NS equations exactly
        (forcing == 0 identically) and is divergence-free."""
        Fx, Fy = self.forcing()
        return Fx == 0 and Fy == 0 and self.divergence() == 0

    def lambdify_all(self, nu_val: float = None, rho_val: float = 1.0,
                      nu_x_val: float = None, nu_y_val: float = None,
                      extra_subs: dict = None):
        """
        Numerically evaluable (numpy) versions of u, v, p and the exact
        forcing (Fx, Fy), each a callable (X, Y, t) -> ndarray. This is
        the bridge from the symbolic MMS engine (WS0.4) to the numerical
        FV solver (WS2): manufactured IC/BC and the source term the
        solver must add to reproduce it exactly.

        Isotropic (default): pass nu_val; forcing uses the single `nu`
        symbol for both directions (unchanged from before).

        K5 (anisotropic): pass nu_x_val, nu_y_val (and leave nu_val
        None) to compute forcing with the anisotropic Laplacian
        nu_x*d2/dx2 + nu_y*d2/dy2.

        extra_subs: additional {symbol: value} substitutions (e.g. K1's
        `epsilon`, or K2's convection-amplitude/frequency symbols).
        """
        subs = {rho: sp.Float(rho_val)}
        if nu_x_val is not None or nu_y_val is not None:
            assert nu_val is None, "pass either nu_val (isotropic) or nu_x_val/nu_y_val (K5), not both"
            subs[nu_x] = sp.Float(nu_x_val)
            subs[nu_y] = sp.Float(nu_y_val)
            Fx_sym, Fy_sym = self.forcing(nu_x_sym=nu_x, nu_y_sym=nu_y)
        else:
            subs[nu] = sp.Float(nu_val)
            Fx_sym, Fy_sym = self.forcing()

        if extra_subs:
            subs.update({k: sp.Float(v) for k, v in extra_subs.items()})

        u_num = sp.lambdify((x, y, t), self.u.subs(subs), "numpy")
        v_num = sp.lambdify((x, y, t), self.v.subs(subs), "numpy")
        p_num = sp.lambdify((x, y, t), self.p.subs(subs), "numpy")
        Fx_num = sp.lambdify((x, y, t), Fx_sym.subs(subs), "numpy")
        Fy_num = sp.lambdify((x, y, t), Fy_sym.subs(subs), "numpy")

        return u_num, v_num, p_num, Fx_num, Fy_num


def base_tgv_field(Uc=1, Vc=0, V0=1, L=1, p0=0):
    """The exact challenge solution as a symbolic MMSField, for verifying
    D1 (constitution v2): its forcing must be identically zero."""
    Uc, Vc, V0, L, p0 = (sp.sympify(a) for a in (Uc, Vc, V0, L, p0))

    f = sp.exp(-2 * nu * t / L**2)
    xt = (x - Uc * t) / L
    yt = (y - Vc * t) / L

    u_ = Uc + V0 * sp.sin(xt) * sp.cos(yt) * f
    v_ = Vc - V0 * sp.cos(xt) * sp.sin(yt) * f
    p_ = p0 + (rho * V0**2 / 4) * f**2 * (sp.cos(2 * xt) + sp.cos(2 * yt))

    return MMSField(u=u_, v=v_, p=p_)


def anisotropic_tgv_field(Uc=1, Vc=0, V0=1, L=1, p0=0):
    """
    K5 knob (constitution v2 §3): the exact solution of the LINEAR
    advection-diffusion equation with ANISOTROPIC diffusion nu_x, nu_y
    (D1's cancellation is unaffected -- it never involves nu at all, so
    the pressure formula is identical to the isotropic case). The decay
    rate becomes exp(-(nu_x+nu_y)*t) instead of exp(-2*nu*t): applying
    nu_x*d2/dx2 + nu_y*d2/dy2 to sin(xt)cos(yt) gives eigenvalue
    -(nu_x+nu_y), vs. -2*nu in the isotropic (nu_x=nu_y=nu) corner,
    where this reduces exactly to base_tgv_field's decay.

    Call `.forcing(nu_x_sym=nu_x, nu_y_sym=nu_y)` (or use
    `.lambdify_all(nu_x_val=..., nu_y_val=...)`) to get the correct
    anisotropic forcing -- using the default isotropic `.forcing()` on
    this field would use the WRONG (single-nu) Laplacian and not
    reproduce zero forcing at the ratio=1 corner.
    """
    Uc, Vc, V0, L, p0 = (sp.sympify(a) for a in (Uc, Vc, V0, L, p0))

    f = sp.exp(-(nu_x + nu_y) * t / L**2)
    xt = (x - Uc * t) / L
    yt = (y - Vc * t) / L

    u_ = Uc + V0 * sp.sin(xt) * sp.cos(yt) * f
    v_ = Vc - V0 * sp.cos(xt) * sp.sin(yt) * f
    p_ = p0 + (rho * V0**2 / 4) * f**2 * (sp.cos(2 * xt) + sp.cos(2 * yt))

    return MMSField(u=u_, v=v_, p=p_)


def k2_time_dependent_convection_field(Uc0=1, amplitude=sp.Symbol("A", real=True),
                                        omega=2, Vc=0, V0=1, L=1, p0=0):
    """
    K2 knob (constitution v2 §3): background convection speed
    Uc(t) = Uc0 + amplitude*sin(omega*t) instead of a constant. The
    traveling-wave phase must track the actual displacement
    X(t) = integral_0^t Uc(t') dt' = Uc0*t - (amplitude/omega)*(cos(omega*t)-1),
    not Uc(t)*t. At amplitude=0 this reduces exactly to base_tgv_field
    (Uc(t) -> Uc0, X(t) -> Uc0*t).

    The resulting field is generally NOT an exact solution of the
    unforced equations even for the fluctuating part (there is now a
    genuine d(Uc)/dt term in the u-momentum equation from the
    time-varying mean flow, on top of whatever nonlinear-term behavior
    holds) -- forcing() is expected to be nonzero for amplitude != 0,
    which is exactly K2 "breaking" the degenerate corner.
    """
    Uc0, Vc, V0, L, p0 = (sp.sympify(a) for a in (Uc0, Vc, V0, L, p0))
    omega = sp.sympify(omega)

    Uc_t = Uc0 + amplitude * sp.sin(omega * t)
    X_t = sp.integrate(Uc_t, t)  # displacement, tracks the time-varying speed

    f = sp.exp(-2 * nu * t / L**2)
    xt = (x - X_t) / L
    yt = (y - Vc * t) / L

    u_ = Uc_t + V0 * sp.sin(xt) * sp.cos(yt) * f
    v_ = Vc - V0 * sp.cos(xt) * sp.sin(yt) * f
    p_ = p0 + (rho * V0**2 / 4) * f**2 * (sp.cos(2 * xt) + sp.cos(2 * yt))

    return MMSField(u=u_, v=v_, p=p_)


def k1_perturbed_field(eps=sp.Symbol("epsilon", real=True), k2=3, Uc=1, Vc=0, V0=1, L=1, p0=0):
    """
    K1 knob (constitution v2 §3): superpose an incommensurate second mode
    of amplitude `eps` onto the base TGV field. At eps=0 this reduces
    exactly to base_tgv_field. For eps != 0 the projected nonlinear term
    generally does NOT vanish, so forcing() != 0 -- that non-vanishing
    forcing is what an MMS solver adds to the RHS to keep the perturbed
    field an EXACT solution despite breaking D1's cancellation.

    k2 is the second mode's wavenumber (k2 != 1 => incommensurate with
    the base mode, so the cross term does not telescope into a pure
    gradient the way the base TGV's self-interaction does).
    """
    base = base_tgv_field(Uc=Uc, Vc=Vc, V0=V0, L=L, p0=p0)

    u2 = eps * sp.sin(k2 * x) * sp.cos(k2 * y)
    v2 = -eps * sp.cos(k2 * x) * sp.sin(k2 * y)

    return MMSField(u=base.u + u2, v=base.v + v2, p=base.p)
