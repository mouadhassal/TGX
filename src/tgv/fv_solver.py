"""
WS2: dense (uncompressed) flux-form finite-volume incompressible NS
solver on a doubly-periodic collocated grid. This is baseline #1 in
§9.2 ("dense 2nd-order FV, explicit -- isolates the TT machinery") and
also the reference implementation that WS2's quantics/TT compression
will later be checked against (A7: flux-form vs point-value FD).

Discretization (2nd-order central, flux-difference form):
  For each momentum component, cell-face values are linear
  interpolations of neighboring cell averages (u_face = 0.5*(u_i+u_{i+1})),
  the convective flux at a face is the product of face-interpolated
  velocities, the diffusive flux at a face is nu times the face-normal
  gradient, and the cell update is the difference of face fluxes divided
  by the cell width -- i.e. an explicit statement of flux conservation,
  not a finite-difference stencil in disguise (A7's spec-compliance
  claim rests on this).

Time integration: explicit RK2 (Heun) for the convective-diffusive
predictor, then a spectral (FFT) pressure projection enforcing
div(u)=0 exactly (to machine precision) on the periodic domain, forming
the pressure field explicitly (Q11).
"""
from dataclasses import dataclass

import numpy as np


def _face_avg(a: np.ndarray, axis: int) -> np.ndarray:
    """Face value at i+1/2 (axis direction), periodic."""
    return 0.5 * (a + np.roll(a, -1, axis=axis))


def _flux_diff(face_vals: np.ndarray, dx: float, axis: int) -> np.ndarray:
    """(F[i+1/2] - F[i-1/2]) / dx, i.e. F[i+1/2]-F at i, minus F at i-1
    (roll +1 to bring i-1/2's value to position i)."""
    return (face_vals - np.roll(face_vals, 1, axis=axis)) / dx


def convection_diffusion_rhs(u, v, dx, dy, nu_x, nu_y=None):
    """Flux-form RHS of the momentum equations, excluding pressure:
    d/dt(u,v) = -div(flux) + nu_x*d2/dx2(u,v) + nu_y*d2/dy2(u,v).
    Returns (rhs_u, rhs_v).

    K5 (anisotropic diffusion): pass distinct nu_x, nu_y. Backward
    compatible with every existing (isotropic) call site: nu_y=None
    defaults to nu_y=nu_x, i.e. passing a single value behaves exactly
    as before.
    """
    if nu_y is None:
        nu_y = nu_x

    u_fx = _face_avg(u, axis=0)   # u at x-faces (i+1/2, j)
    v_fx = _face_avg(v, axis=0)   # v at x-faces (i+1/2, j)
    u_fy = _face_avg(u, axis=1)   # u at y-faces (i, j+1/2)
    v_fy = _face_avg(v, axis=1)   # v at y-faces (i, j+1/2)

    du_dx_face = (np.roll(u, -1, axis=0) - u) / dx
    dv_dx_face = (np.roll(v, -1, axis=0) - v) / dx
    du_dy_face = (np.roll(u, -1, axis=1) - u) / dy
    dv_dy_face = (np.roll(v, -1, axis=1) - v) / dy

    # u-momentum fluxes
    Fx_u = u_fx * u_fx - nu_x * du_dx_face
    Fy_u = u_fy * v_fy - nu_y * du_dy_face
    rhs_u = -_flux_diff(Fx_u, dx, axis=0) - _flux_diff(Fy_u, dy, axis=1)

    # v-momentum fluxes
    Fx_v = u_fx * v_fx - nu_x * dv_dx_face
    Fy_v = v_fy * v_fy - nu_y * dv_dy_face
    rhs_v = -_flux_diff(Fx_v, dx, axis=0) - _flux_diff(Fy_v, dy, axis=1)

    return rhs_u, rhs_v


def divergence(u, v, dx, dy):
    """Cell-centered divergence via central difference of face-averaged
    normal velocity (consistent with the flux-form discretization).

    AUDIT.md N1: this operator's Fourier symbol is i*sin(k*dx)/dx (see
    project_divergence_free's derivation), which is IDENTICALLY ZERO at
    the Nyquist mode -- the same symbol project_divergence_free inverts
    and explicitly pins to zero. This diagnostic therefore CANNOT see
    checkerboard content by construction; a max_div computed with THIS
    function alone proves less than it appears to. Use
    `divergence_spectral` (continuum symbol, nonzero at Nyquist) as an
    independent cross-check wherever divergence-free-ness is being
    used as evidence, not just a sanity print.
    """
    dudx = (_face_avg(u, axis=0) - np.roll(_face_avg(u, axis=0), 1, axis=0)) / dx
    dvdy = (_face_avg(v, axis=1) - np.roll(_face_avg(v, axis=1), 1, axis=1)) / dy
    return dudx + dvdy


def divergence_spectral(u, v, dx, dy):
    """
    AUDIT.md N1: INDEPENDENT divergence diagnostic using a ONE-SIDED
    (forward) difference, symbol (exp(i*k*dx)-1)/dx -- NONZERO at
    Nyquist (k*dx=pi gives -2/dx, not 0) -- unlike `divergence`'s
    compact-stencil symbol, which vanishes there and is blind to
    checkerboard content.

    NOTE on a real pitfall found while building this: the "obvious"
    independent check -- pure continuum spectral differentiation via
    ik and FFT -- is ALSO blind to Nyquist content on an even-length
    real grid, for a different and easy-to-miss reason: a real signal's
    Nyquist Fourier coefficient is self-conjugate, so ik times it is
    PURELY IMAGINARY, and taking the real part (needed to get a real
    derivative back) silently discards exactly the content this
    diagnostic exists to catch. That version was tried first here, and
    it passed all its own tests by returning 0.0 -- the same blind spot
    it was meant to detect, just relocated. The one-sided stencil below
    has no such self-conjugacy issue and was verified against a pure
    Nyquist checkerboard input to actually produce a large, correct
    signal (see tests/test_n1_divergence_diagnostics.py).
    """
    du_dx = (np.roll(u, -1, axis=0) - u) / dx
    dv_dy = (np.roll(v, -1, axis=1) - v) / dy
    return du_dx + dv_dy


def project_divergence_free(u_star, v_star, dx, dy, dt, rho=1.0):
    """
    Spectral (FFT) pressure projection on the periodic domain, using the
    DISCRETE Fourier symbol of the actual compact central-difference
    divergence stencil used by `divergence()` -- NOT the continuum
    symbol (ik, -k^2). Using the continuum symbol here would be
    inconsistent with the discrete divergence operator and leaves the
    Nyquist ("checkerboard") mode uncorrected, an odd-even decoupling
    artifact intrinsic to collocated central differencing. Matching the
    symbols exactly makes div(u_new) vanish, per the discrete operator,
    at every mode except the two structurally unconstrained ones
    (DC, k=0, and Nyquist, where the compact stencil's symbol is
    identically zero) -- those are pinned to zero explicitly.
    """
    N, M = u_star.shape
    div_star = divergence(u_star, v_star, dx, dy)
    rhs_hat = np.fft.fft2(div_star / dt)

    # discrete symbol of D(phi)[i] = (phi[i+1]-phi[i-1])/(2*dx) is
    # i*sin(kappa*dx)/dx (verified in fastforward.py / tests); the
    # discrete Laplacian this projection must invert is D_x^2 + D_y^2,
    # whose symbol is -(sin(kappa_x*dx)/dx)^2 - (sin(kappa_y*dy)/dy)^2.
    kx = 2 * np.pi * np.fft.fftfreq(N, d=dx) * dx  # kappa_x*dx
    ky = 2 * np.pi * np.fft.fftfreq(M, d=dy) * dy  # kappa_y*dy
    KX, KY = np.meshgrid(kx, ky, indexing="ij")

    symbol_dx = 1j * np.sin(KX) / dx
    symbol_dy = 1j * np.sin(KY) / dy
    lap_op = -(np.sin(KX) / dx) ** 2 - (np.sin(KY) / dy) ** 2

    singular = np.abs(lap_op) < 1e-12 * (1.0 / dx ** 2)
    lap_op_safe = np.where(singular, 1.0, lap_op)

    phi_hat = rhs_hat / lap_op_safe
    phi_hat[singular] = 0.0  # DC + Nyquist: structurally unconstrained, pin to 0
    phi = np.real(np.fft.ifft2(phi_hat))

    dphidx = np.real(np.fft.ifft2(symbol_dx * phi_hat))
    dphidy = np.real(np.fft.ifft2(symbol_dy * phi_hat))

    u = u_star - dt * dphidx
    v = v_star - dt * dphidy
    p = rho * phi
    return u, v, p


@dataclass
class StepDiagnostics:
    max_div: float


def step(u, v, dt, dx, dy, nu, nu_y=None, rho=1.0, forcing=None, t=None):
    """One explicit RK2 (Heun) predictor-corrector step for the
    convective-diffusive part, followed by pressure projection.

    K5: pass nu_y != nu (nu is then used as nu_x) for anisotropic
    diffusion; defaults to isotropic (nu_y=None -> nu_y=nu), unchanged
    from every existing call site.

    `forcing`, if given, is a callable (t) -> (Fx, Fy) arrays added to
    the momentum RHS -- the MMS forcing hook (WS0.4 engine)."""
    def rhs_with_forcing(u_, v_, t_):
        ru, rv = convection_diffusion_rhs(u_, v_, dx, dy, nu, nu_y)
        if forcing is not None:
            Fx, Fy = forcing(t_)
            ru = ru + Fx
            rv = rv + Fy
        return ru, rv

    ru0, rv0 = rhs_with_forcing(u, v, t)
    u_pred_star = u + dt * ru0
    v_pred_star = v + dt * rv0

    # AUDIT-3 §1: project the PREDICTOR stage too (incremental pressure
    # correction), not only the final combined stage. An unprojected
    # predictor is the standard cause of a fractional-step scheme
    # degrading to 1st order in time regardless of the RK2 stage order
    # -- confirmed directly (not via Richardson extrapolation, which an
    # earlier pass misdiagnosed as floor contamination of a 2nd-order
    # method): at fixed, fine N (spatial error negligible), halving dt
    # exactly halved the error (ratio ~2.0, not ~4.0) -- unambiguous 1st
    # order. Projecting the predictor removes it.
    u_pred, v_pred, _ = project_divergence_free(u_pred_star, v_pred_star, dx, dy, dt, rho)

    ru1, rv1 = rhs_with_forcing(u_pred, v_pred, t + dt if t is not None else None)
    u_star = u + 0.5 * dt * (ru0 + ru1)
    v_star = v + 0.5 * dt * (rv0 + rv1)

    u_new, v_new, p_new = project_divergence_free(u_star, v_star, dx, dy, dt, rho)
    diag = StepDiagnostics(max_div=float(np.max(np.abs(divergence(u_new, v_new, dx, dy)))))
    return u_new, v_new, p_new, diag


def run(u0, v0, dx, dy, nu, dt, n_steps, nu_y=None, rho=1.0, forcing=None, t0=0.0):
    """March n_steps of size dt from (u0, v0) at time t0. Returns final
    (u, v, p) and the list of per-step max-divergence diagnostics.
    K5: pass nu_y for anisotropic diffusion (nu is then nu_x)."""
    u, v = u0.copy(), v0.copy()
    p = None
    diags = []
    t = t0
    for _ in range(n_steps):
        u, v, p, diag = step(u, v, dt, dx, dy, nu, nu_y=nu_y, rho=rho, forcing=forcing, t=t)
        diags.append(diag)
        t += dt
    return u, v, p, diags
