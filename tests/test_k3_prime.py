"""
K3' (AUDIT.md S1): reinstates a D3-stressing knob after K3 (MMS-forced,
prescribed spectral slope) was cut by its own kill criterion. Pre-
registered criterion for K3' (Rule 4): none needed -- the construction
is exact by linearity for any mode set, with zero forcing, so there is
no "measuring the forcing instead of the flow" failure mode to guard
against in the first place.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import sympy as sp

from tgv.k3_prime import multi_mode_field, verify_exact_symbolically, occupied_subspace_kappa_multimode
from tgv.quantics import quantics_reshape_2d, tt_ranks


def test_k3_prime_two_mode_case_is_symbolically_exact():
    residual = verify_exact_symbolically(ks=[1, 3], amplitudes=[1.0, 0.4], nu_val=0.05)
    assert residual == 0, residual


def test_k3_prime_five_mode_case_is_symbolically_exact():
    residual = verify_exact_symbolically(ks=[1, 2, 3, 5, 7], amplitudes=[1.0, 0.5, 0.4, 0.3, 0.2],
                                          phases=[0, 0.3, 0.7, 1.1, 1.9], nu_val=0.05)
    assert residual == 0, residual


def test_k3_prime_rank_grows_with_mode_count_zero_forcing():
    """The actual D3 stress test: chi(m) measured on the EXACT field
    (no solver, no forcing) as the number of superposed incommensurate
    modes grows.

    AUDIT-2 §6 minor: the originally-reported chi(m)=[4,8,16,19] was
    ITSELF tolerance-limited at m=8 -- tol=1e-9 gave 19, but tol=1e-12
    (and 1e-14, identical) gives the converged value 24. Fixed by using
    tol=1e-12 and checking stability against an even tighter tolerance
    explicitly, rather than reporting whatever a single, unchecked
    tolerance happens to produce.

    Also note explicitly (per the audit): chi(m=1)=4, one below D3's
    own chi<=5 bound for the FULL TGV field. Consistent, not a
    contradiction -- this construction is PURE FLUCTUATION (no added
    Uc constant mean-flow term), so it is missing the one extra rank
    contribution the full field's constant term adds.
    """
    N = 2 ** 8
    domain_length = 2 * np.pi
    xs = np.linspace(0, domain_length, N, endpoint=False)
    X, Y = np.meshgrid(xs, xs, indexing="ij")

    rng = np.random.default_rng(0)
    all_ks = [1, 3, 5, 7, 11, 13, 17, 19]  # incommensurate wavenumbers

    chis = []
    for m in (1, 2, 4, 8):
        ks = all_ks[:m]
        amps = rng.uniform(0.3, 1.0, size=m)
        field = multi_mode_field(X, Y, t=0.7, ks=ks, amplitudes=amps, nu=0.05)
        chi = max(tt_ranks(quantics_reshape_2d(field), tol=1e-12))
        chi_tighter = max(tt_ranks(quantics_reshape_2d(field), tol=1e-14))
        assert chi == chi_tighter, (m, chi, chi_tighter)  # confirms tolerance-converged
        chis.append(chi)

    print(f"chi(m) for m in (1,2,4,8), tolerance-converged: {chis}", file=sys.stderr)

    assert chis == [4, 8, 16, 24], chis         # exact, tolerance-converged values
    assert chis == sorted(chis), chis           # non-decreasing in m
    assert chis[-1] > chis[0], chis             # strictly grows somewhere
    assert chis[0] == 4, chis                   # one below D3's chi<=5 -- see docstring (no mean-flow term)


def test_k3_prime_reduces_to_single_mode_bound_at_m_equals_1():
    """m=1 corner sanity check: a lone mode has the same rank structure
    as the base TGV field's oscillatory part (rank <= 5 territory)."""
    N = 2 ** 8
    xs = np.linspace(0, 2 * np.pi, N, endpoint=False)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    field = multi_mode_field(X, Y, t=0.3, ks=[1], amplitudes=[1.0], nu=0.05)
    chi = max(tt_ranks(quantics_reshape_2d(field), tol=1e-9))
    assert chi <= 5, chi


def test_k3_prime_occupied_subspace_kappa_grows_with_mode_count():
    """
    AUDIT-2 §4: K5 (grid anisotropy) CANNOT break D5 -- the TGV occupies
    ONE mode, so the operator restricted to it is a 1x1 block, condition
    number exactly 1 for every grid, every aspect ratio (K5's own test
    already shows this: occupied kappa 1.0000->1.0008, a metric that
    cannot move, not evidence "surviving a stress"). K3' genuinely CAN:
    with m incommensurate modes, occupied kappa = k_max^2/k_min^2,
    tunable by construction. This is the correct D5-breaking knob.
    """
    all_ks = [1, 3, 5, 7, 11, 13, 17, 19]
    kappas = []
    for m in (1, 2, 4, 8):
        kappas.append(occupied_subspace_kappa_multimode(all_ks[:m]))

    print(f"occupied-subspace kappa(m) for m in (1,2,4,8): {kappas}", file=sys.stderr)

    assert kappas[0] == 1.0, kappas              # single mode: matches D5's claim exactly
    assert kappas == sorted(kappas), kappas       # monotonically grows with m
    assert kappas[-1] > 100, kappas               # genuinely tunable, unlike K5 (max ~1.0008)


def test_k3_prime_kappa_at_fixed_chi_orthogonal_subknob():
    """
    AUDIT-3 §2: the original 'kappa(m) and chi(m) grow together from
    the same mode sets' test CONFOUNDS D3 and D5 -- a knob that moves
    two degeneracies at once cannot attribute a method's cost increase
    to either one. Fixed with two independent sub-knobs (the audit's
    suggested structure; the specific mode sets found here give an even
    cleaner separation than the audit's own illustrative numbers).

    Sub-knob 1: vary SPREAD at fixed COUNT (always 2 modes, {1, k}).
    chi stays PERFECTLY fixed at 8 while kappa = k^2 ranges from 4 to
    225 (56x) as k goes 2->15 -- a clean kappa-only signal.
    """
    N = 2 ** 8
    xs = np.linspace(0, 2 * np.pi, N, endpoint=False)
    X, Y = np.meshgrid(xs, xs, indexing="ij")

    ks_list = [[1, 2], [1, 3], [1, 5], [1, 7], [1, 10], [1, 15]]
    chis, kappas = [], []
    for ks in ks_list:
        field = multi_mode_field(X, Y, t=0.7, ks=ks, amplitudes=[1.0] * len(ks), nu=0.05)
        chis.append(max(tt_ranks(quantics_reshape_2d(field), tol=1e-12)))
        kappas.append(occupied_subspace_kappa_multimode(ks))

    print(f"kappa-at-fixed-chi: ks={ks_list}, chi={chis}, kappa={kappas}", file=sys.stderr)

    assert len(set(chis)) == 1, chis            # chi PERFECTLY fixed
    assert kappas == sorted(kappas), kappas       # kappa monotonically grows
    assert kappas[-1] / kappas[0] > 50, kappas    # genuinely tunable, >1 order


def test_k3_prime_chi_at_fixed_kappa_orthogonal_subknob():
    """
    Sub-knob 2 (AUDIT-3 §2): vary COUNT at fixed SPREAD (endpoints
    always k=1 and k=10, so kappa=100 always). chi grows 8->12->16->24
    (3x) as intermediate modes are added -- a clean chi-only signal,
    orthogonal to sub-knob 1 above. Together, the two sub-knobs let a
    cost increase be attributed to D3 (chi) or D5 (kappa) individually,
    which the original confounded single knob could not do.
    """
    N = 2 ** 8
    xs = np.linspace(0, 2 * np.pi, N, endpoint=False)
    X, Y = np.meshgrid(xs, xs, indexing="ij")

    ks_list = [[1, 10], [1, 3, 10], [1, 3, 5, 10], [1, 2, 3, 5, 7, 10]]
    chis, kappas = [], []
    for ks in ks_list:
        field = multi_mode_field(X, Y, t=0.7, ks=ks, amplitudes=[1.0] * len(ks), nu=0.05)
        chis.append(max(tt_ranks(quantics_reshape_2d(field), tol=1e-12)))
        kappas.append(occupied_subspace_kappa_multimode(ks))

    print(f"chi-at-fixed-kappa: ks={ks_list}, chi={chis}, kappa={kappas}", file=sys.stderr)

    assert len(set(kappas)) == 1, kappas         # kappa PERFECTLY fixed
    assert chis == sorted(chis), chis            # chi monotonically grows
    assert chis[-1] / chis[0] >= 3, chis          # genuinely tunable
