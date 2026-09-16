"""
K2 knob (constitution v2 §3): Uc(t) time-dependent breaks D2's
fast-forwarding, because the generator is no longer time-independent, so
a single QFT/diagonal-phase/inverse-QFT application (valid only for a
constant-coefficient generator) no longer reaches the correct state at
time T -- genuine time-marching (or a Magnus-type multi-interval
expansion) is required. This is WS3's first knob, "near-free" per the
constitution since it only changes Uc from a constant to a function.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from tgv.fastforward import advection_diffusion_operator, analytic_dispersion, fast_forward_apply


def true_time_varying_evolution(u0, N, Uc_of_t, nu, domain_length, T, n_substeps=2000):
    """Ground truth for a TIME-DEPENDENT-coefficient advection-diffusion
    equation: RK4 with many small substeps, rebuilding the (still
    circulant-in-space, but now time-varying) operator at each substep.
    This is exactly the genuine time-marching K2 forces."""
    u = u0.copy().astype(complex)
    dt = T / n_substeps
    t = 0.0
    for _ in range(n_substeps):
        def deriv(u_, t_):
            A = advection_diffusion_operator(N, Uc_of_t(t_), nu, domain_length)
            return A @ u_

        k1 = deriv(u, t)
        k2 = deriv(u + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = deriv(u + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = deriv(u + dt * k3, t + dt)
        u = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
    return u


def test_k2_naive_fastforward_matches_truth_when_uc_is_actually_constant():
    """Sanity check: with Uc(t) == const, the naive single-shot
    fast-forward (built for a time-INDEPENDENT generator) must still
    match genuine time-marching -- this is D2 itself, re-confirmed here
    as the lambda=0 corner of the K2 knob."""
    N = 16
    Uc_const, nu, L = 1.0, 0.05, 2 * np.pi
    rng = np.random.default_rng(0)
    u0 = rng.normal(size=N)

    T = 2.0
    truth = true_time_varying_evolution(u0, N, lambda t: Uc_const, nu, L, T, n_substeps=200)

    eig = analytic_dispersion(N, Uc_const, nu, L)
    naive = fast_forward_apply(u0, eig, T)

    assert np.allclose(naive.real, truth.real, atol=1e-5), np.max(np.abs(naive.real - truth.real))


def test_k2_time_dependent_uc_breaks_naive_fastforward():
    """With Uc(t) = 1 + 0.5*sin(2t) (K2 > 0), the naive fast-forward
    built from ANY fixed Uc (e.g. the time-average) must diverge
    materially from genuine time-marching -- confirming D2's shortcut is
    unavailable once the generator is time-dependent."""
    N = 16
    nu, L = 0.05, 2 * np.pi
    rng = np.random.default_rng(0)
    u0 = rng.normal(size=N)

    T = 2.0
    Uc_of_t = lambda t: 1.0 + 0.5 * np.sin(2.0 * t)
    truth = true_time_varying_evolution(u0, N, Uc_of_t, nu, L, T, n_substeps=400)

    Uc_avg = 1.0  # time-average of 1 + 0.5*sin(2t) over a period
    eig = analytic_dispersion(N, Uc_avg, nu, L)
    naive = fast_forward_apply(u0, eig, T)

    err = np.max(np.abs(naive.real - truth.real))
    ref_scale = np.max(np.abs(truth.real))
    assert err > 0.05 * ref_scale, (err, ref_scale)  # not a small correction
