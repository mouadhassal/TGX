"""
D5: verify worst-case kappa grows like O(N^2) (diffusion-dominated,
standard result) while the occupied-subspace kappa stays exactly 1,
independent of N -- the discriminator between algorithm classes.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from tgv.conditioning import worst_case_kappa, occupied_subspace_kappa


def test_d5_worst_case_kappa_grows_quadratically_with_n():
    Uc, nu, L = 1.0, 2 * np.pi / 100, 2 * np.pi
    Ns = [32, 64, 128, 256]
    kappas = [worst_case_kappa(N, Uc, nu, L) for N in Ns]

    # doubling N should roughly quadruple kappa in the diffusion-dominated
    # regime (kappa ~ N^2); check the measured exponent is close to 2
    log_N = np.log(Ns)
    log_k = np.log(kappas)
    slope = np.polyfit(log_N, log_k, 1)[0]
    assert 1.7 <= slope <= 2.3, (kappas, slope)


def test_d5_occupied_subspace_kappa_is_exactly_one_independent_of_n():
    Uc, nu, L = 1.0, 2 * np.pi / 100, 2 * np.pi
    for N in (32, 64, 128, 256, 1024):
        kappa_occ = occupied_subspace_kappa(N, Uc, nu, L, occupied_k=1)
        assert kappa_occ == 1.0, (N, kappa_occ)


def test_d5_gap_between_worst_case_and_occupied_widens_with_n():
    """This is the actual discriminator: the GAP between what a
    worst-case method pays and what a solution-adaptive method pays
    grows with N, even though the physical solution never changes."""
    Uc, nu, L = 1.0, 2 * np.pi / 100, 2 * np.pi
    gaps = []
    for N in (64, 256, 1024):
        k_worst = worst_case_kappa(N, Uc, nu, L)
        k_occ = occupied_subspace_kappa(N, Uc, nu, L)
        gaps.append(k_worst / k_occ)
    assert gaps == sorted(gaps)
    assert gaps[-1] / gaps[0] > 100  # widens substantially over a 16x grid
