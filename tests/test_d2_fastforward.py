"""
D2: verify circulant structure, DFT diagonalization, and time-independent
cost of fast-forwarding, numerically (constitution v2 §2 D2 / A2).
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import time
import numpy as np
from scipy.linalg import expm

from tgv.fastforward import (
    advection_diffusion_operator,
    is_circulant,
    dft_eigenvalues,
    analytic_dispersion,
    fast_forward_apply,
)


def test_d2_operator_is_circulant():
    A = advection_diffusion_operator(N=32, Uc=1.0, nu=0.05, domain_length=2 * np.pi)
    assert is_circulant(A)


def test_d2_dft_eigenvalues_match_dispersion_relation():
    N = 32
    A = advection_diffusion_operator(N=N, Uc=1.0, nu=0.05, domain_length=2 * np.pi)
    eig_dft = dft_eigenvalues(A)
    eig_analytic = analytic_dispersion(N=N, Uc=1.0, nu=0.05, domain_length=2 * np.pi)
    assert np.allclose(eig_dft, eig_analytic, atol=1e-10)


def test_d2_fast_forward_matches_matrix_exponential_for_arbitrary_t():
    N = 16
    Uc, nu, L = 1.0, 0.05, 2 * np.pi
    A = advection_diffusion_operator(N=N, Uc=Uc, nu=nu, domain_length=L)
    eig = analytic_dispersion(N=N, Uc=Uc, nu=nu, domain_length=L)

    rng = np.random.default_rng(0)
    u0 = rng.normal(size=N)

    for t in (0.001, 1.0, 10.0, 1000.0):
        expected = expm(t * A) @ u0
        got = fast_forward_apply(u0, eig, t)
        assert np.allclose(got.real, expected, atol=1e-6), (t, np.max(np.abs(got.real - expected)))


def test_d2_fast_forward_cost_is_independent_of_t():
    """The whole point of D2: wall-clock cost of reaching time T must not
    grow with T (no CFL / no time-marching), unlike a naive stepper."""
    N = 4096
    Uc, nu, L = 1.0, 2 * np.pi / 100, 2 * np.pi
    eig = analytic_dispersion(N=N, Uc=Uc, nu=nu, domain_length=L)
    rng = np.random.default_rng(0)
    u0 = rng.normal(size=N)

    times_measured = []
    for t in (1.0, 10.0, 1e6, 1e12):
        t0 = time.perf_counter()
        fast_forward_apply(u0, eig, t)
        t1 = time.perf_counter()
        times_measured.append(t1 - t0)

    # cost must not scale with T: max/min ratio should be O(1), not
    # anywhere near linear in T's ~1e12x range
    ratio = max(times_measured) / min(times_measured)
    assert ratio < 20, times_measured
