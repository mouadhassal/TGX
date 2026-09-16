"""
D5 (constitution v2 §2, "effective condition number is 1 on the occupied
subspace"): the solution occupies a single wavenumber pair. A method
whose cost tracks the SOLUTION (Krylov restricted to the occupied
subspace, low-rank, adaptive) sees effective kappa = O(1); a method whose
cost tracks the WORST-CASE spectrum of the discrete operator pays the
usual kappa ~ N^2 diffusion penalty.

Reuses the exact discrete operator spectrum from fastforward.py so this
is not a separate, potentially-inconsistent derivation.
"""
import numpy as np

from .fastforward import analytic_dispersion


def worst_case_kappa(N: int, Uc: float, nu: float, domain_length: float) -> float:
    """
    kappa = max|lambda_k| / min|lambda_k| over all nonzero modes k != 0
    of the discrete advection-diffusion operator -- the condition number
    a worst-case (operator-spectrum-tracking) linear solve pays.
    """
    lam = analytic_dispersion(N, Uc, nu, domain_length)
    mags = np.abs(lam)
    mags = mags[mags > 1e-12]  # drop the k=0 (constant/mean-flow) mode
    return float(np.max(mags) / np.min(mags))


def anisotropic_dispersion_2d(Nx: int, Ny: int, Uc: float, Vc: float, nu: float,
                               domain_length: float):
    """
    K5 knob (constitution v2 §3): grid anisotropy (Nx != Ny, i.e.
    dx != dy), NOT viscosity anisotropy. nu stays a single ISOTROPIC
    value -- the continuum PDE and its exact solution are unchanged, so
    the reference used to judge any solver run under this knob stays
    exactly valid (AUDIT.md B1: an earlier version of this module used
    nu_x != nu_y, which changes the PDE the solver integrates -- any
    error that produced measured a modeling mismatch, not conditioning,
    and has been removed).

    At Nx=Ny (dx=dy) this reduces to the isotropic case. Returns the 2D
    eigenvalue array lambda(kx, ky) on the (possibly rectangular) grid,
    reusing the exact 1D dispersion relation from fastforward.py,
    evaluated at each axis's OWN resolution (hence own dx) but the SAME
    nu -- so the SAME continuum wavenumber k=1 gets a different
    discrete argument k*dx in each direction once Nx != Ny.
    """
    lam_x = analytic_dispersion(Nx, Uc, nu, domain_length)
    lam_y = analytic_dispersion(Ny, Vc, nu, domain_length)
    LX, LY = np.meshgrid(lam_x, lam_y, indexing="ij")
    return LX + LY


def anisotropic_kappa(Nx: int, Ny: int, Uc: float, Vc: float, nu: float,
                       domain_length: float) -> float:
    """Worst-case kappa of the 2D grid-anisotropic operator: as Nx/Ny
    moves away from 1, the discrete Nyquist limits differ per axis and
    the pooled spectrum spreads -- this is K5's mechanism for breaking
    D5's worst-case number, at fixed isotropic nu (unchanged PDE)."""
    lam = anisotropic_dispersion_2d(Nx, Ny, Uc, Vc, nu, domain_length)
    mags = np.abs(lam)
    mags = mags[mags > 1e-12]
    return float(np.max(mags) / np.min(mags))


def occupied_subspace_kappa_anisotropic(Nx: int, Ny: int, Uc: float, Vc: float, nu: float,
                                         domain_length: float) -> float:
    """
    CORRECTED CLAIM (AUDIT-2 §4): this function does NOT, and cannot,
    demonstrate K5 "breaking D5." The TGV occupies a SINGLE mode; the
    operator restricted to a one-mode occupied subspace is a 1x1 block,
    whose condition number is exactly 1 by definition, for every grid,
    every aspect ratio, every Re -- no grid deformation can move it.
    Measured: 1.0000 -> 1.0008 over a 10x aspect-ratio range, i.e.
    flat, not "surviving a stress" (an earlier, over-read framing of
    this same number, corrected in test_k5_anisotropy.py and
    STATUS.md). The tiny residual movement is a finite-grid discretization
    artifact of comparing the SAME continuum mode resolved on two
    different grids, not a conditioning effect. Retained here as
    documentation of that fact, and for the (now honestly framed)
    contrast against `anisotropic_kappa`'s worst-case number.

    THE KNOB THAT ACTUALLY BREAKS D5 IS K3' (k3_prime.py,
    `occupied_subspace_kappa_multimode`): with m incommensurate modes,
    the occupied subspace is genuinely m-dimensional and its condition
    number k_max^2/k_min^2 is tunable by construction. Use that for any
    claim requiring occupied-subspace kappa to move.

    K5's own genuine, honest contribution is a WALL-CLOCK cost, not a
    conditioning result: the explicit diffusion-stability timestep
    shrinks with the aspect ratio (dt ~ min(dx,dy)^2/nu) -- see
    test_k5_solver_grounded.py's dt-shrinkage test, which stands on its
    own and does not need this function to support it.
    """
    lam = anisotropic_dispersion_2d(Nx, Ny, Uc, Vc, nu, domain_length)
    mx = np.fft.fftfreq(Nx, d=1.0 / Nx)
    my = np.fft.fftfreq(Ny, d=1.0 / Ny)
    idx_x1 = np.argmin(np.abs(mx - 1))  # kx=1, ky=0
    idx_y1 = np.argmin(np.abs(my - 1))  # ky=1, kx=0
    lam_10 = lam[idx_x1, 0]
    lam_01 = lam[0, idx_y1]
    mag_10, mag_01 = abs(lam_10.real), abs(lam_01.real)
    lo, hi = min(mag_10, mag_01), max(mag_10, mag_01)
    return float(hi / lo) if lo > 1e-14 else float("inf")


def occupied_subspace_kappa(N: int, Uc: float, nu: float, domain_length: float,
                             occupied_k: int = 1) -> float:
    """
    kappa restricted to the single occupied wavenumber: since the exact
    TGV solution lives entirely at |k| = occupied_k (plus the k=0 mean
    flow, which any method resolves exactly), a solution-adaptive solver
    only ever needs the condition number of that 1-2 dimensional
    subspace, which is trivially O(1) independent of N.
    """
    dx = domain_length / N
    m = np.fft.fftfreq(N, d=1.0 / N)
    idx = np.argmin(np.abs(m - occupied_k))
    lam = analytic_dispersion(N, Uc, nu, domain_length)
    # single occupied mode vs. the k=0 mean-flow mode it is measured
    # against; ratio of a mode to itself is 1 by construction -- the
    # content of the claim is that NO other mode needs to be resolved.
    return float(np.abs(lam[idx]) / np.abs(lam[idx]))
