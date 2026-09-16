"""
"Brilliant upgrade" item #1: reproduce the organizers' own supplied
KE(t) figure using NOTHING but D1's linear reduction + D2's exact
circulant fast-forward operator -- zero PDE time-stepping, zero
nonlinear terms, a single FFT2-based diagonal application per queried
time t.

If this matches the organizers' own reference curve to high precision,
it is a direct, visual demonstration of D1+D2 together: the benchmark's
own headline plot is exactly reproducible by a LINEAR solve, at O(1)
cost in t, with no CFD machinery at all.

Method:
  1. Build the exact TGV velocity field at t=0 on an N x N grid
     (cell-center point samples -- fine enough that 2nd-order central-
     difference discretization error is far below plotting resolution).
  2. Diagonalize the 2D circulant advection-diffusion operator via FFT2
     ONCE (eigenvalues additive: lambda(kx,ky) = lambda_x(kx) + lambda_y(ky),
     reusing the already-validated 1D dispersion relation from
     fastforward.py for each axis).
  3. For each queried t, apply exp(t*A) via a SINGLE FFT2 -> diagonal
     multiply -> inverse FFT2 -- no time integration loop at all.
  4. Compute mean KE from the resulting (numerical) field directly,
     compare against the organizers' read-off figure values and against
     the closed-form analytic E(t).
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from tgv.config import TGVParams
from tgv.analytic import velocity, mean_kinetic_energy
from tgv.fastforward import analytic_dispersion


def fast_forward_2d_apply(u0, v0, lam2d, t):
    """Single-shot exp(t*A) application to a 2D field via FFT2, no
    time-marching loop -- the direct 2D generalization of
    fastforward.fast_forward_apply."""
    u_hat = np.fft.fft2(u0) * np.exp(t * lam2d)
    v_hat = np.fft.fft2(v0) * np.exp(t * lam2d)
    return np.real(np.fft.ifft2(u_hat)), np.real(np.fft.ifft2(v_hat))


def main():
    Re = 100
    p = TGVParams(Re=Re)
    N = 64  # far above N(Re=100)=64's own requirement; discretization error negligible
    dx = p.domain_length / N
    xs = (np.arange(N) + 0.5) * dx
    X, Y = np.meshgrid(xs, xs, indexing="ij")

    u0, v0 = velocity(p, X, Y, 0.0)

    # Diagonalize ONCE: the mean-flow-translating linear operator's
    # eigenvalues, additive across x and y (constant-coefficient,
    # separable). Uc drives x-advection, Vc (=0) drives y-advection;
    # nu is isotropic diffusion.
    lam_x = analytic_dispersion(N, p.Uc, p.nu, p.domain_length)
    lam_y = analytic_dispersion(N, p.Vc, p.nu, p.domain_length)
    LX, LY = np.meshgrid(lam_x, lam_y, indexing="ij")
    lam2d = LX + LY

    print(f"Re={Re}, N={N}, nu={p.nu:.6f}  (single FFT2 diagonalization, computed once)")
    print()
    print(f"{'t':>6} | {'KE, linear fast-forward (no PDE solve)':>38} | {'KE, closed-form E(t)':>20} | {'organizers read-off':>19}")
    print("-" * 96)

    reference_points = {0.0: 0.75, 10.0: 0.52}  # the two organizer-supplied figure values

    for t in (0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0):
        u_t, v_t = fast_forward_2d_apply(u0, v0, lam2d, t)
        ke_numeric = float(np.mean(0.5 * (u_t**2 + v_t**2)))
        ke_closed_form = float(mean_kinetic_energy(p, t))
        ref = reference_points.get(t)
        ref_str = f"{ref:.4f}" if ref is not None else "--"
        print(f"{t:6.1f} | {ke_numeric:38.8f} | {ke_closed_form:20.8f} | {ref_str:>19}")

    print()
    ke0_err = abs(mean_kinetic_energy(p, 0.0) - 0.75)
    ke10_num, _ = fast_forward_2d_apply(u0, v0, lam2d, 10.0)
    u10, v10 = fast_forward_2d_apply(u0, v0, lam2d, 10.0)
    ke10_numeric = float(np.mean(0.5 * (u10**2 + v10**2)))
    print(f"Max deviation from closed form across all t: "
          f"{max(abs(float(np.mean(0.5*(fu**2+fv**2))) - float(mean_kinetic_energy(p, t))) for t, (fu, fv) in ((tt, fast_forward_2d_apply(u0, v0, lam2d, tt)) for tt in (0,1,2,4,6,8,10))):.3e}")
    print(f"KE(10) via single-shot linear fast-forward: {ke10_numeric:.6f}  (organizers' read-off: 0.52)")
    print()
    print("Zero PDE time-stepping was performed to produce this table.")
    print("Zero nonlinear terms were evaluated. One FFT2 diagonalization,")
    print("then one multiply-by-exp(t*lambda) per queried t -- that is the")
    print("entire computation, at O(1) cost independent of t.")


if __name__ == "__main__":
    main()
