"""
AUDIT-2 §2: verify numerically, before claiming it, that the one-ancilla
dilation's angle computation factorizes into a fixed circuit with
classically precomputed angles -- no quantum arithmetic (square, scale,
exponentiate, arccos) needed at runtime.

Algebra (audit's derivation): with k = sum_j 2^j*b_j and b_j^2=b_j,

    k^2 = sum_j 4^j*b_j + 2*sum_{j<l} 2^(j+l)*b_j*b_l

    e^{-a*k^2} = prod_j e^{-a*4^j*b_j} * prod_{j<l} e^{-2*a*2^(j+l)*b_j*b_l}

Each factor is 1 when its controlling bit(s) are 0 and a classical
constant when they are 1 -- i.e. n singly-controlled and n(n-1)/2
doubly-controlled R_y rotations with COMPILE-TIME angles, composing
via block-encoding (post-selecting all factors yields exactly D|psi>
with success probability ||D|psi>||^2/|||psi>||^2, unchanged).
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np


def _direct_diagonal(n_bits, a):
    k = np.arange(2 ** n_bits, dtype=float)
    return np.exp(-a * k ** 2)


def _product_form_diagonal(n_bits, a):
    """The audit's factorized construction: n singly-controlled +
    n(n-1)/2 doubly-controlled classical-angle factors, composed by
    elementwise product over the 2^n basis states."""
    N = 2 ** n_bits
    k = np.arange(N)
    bits = [(k >> j) & 1 for j in range(n_bits)]

    diag = np.ones(N)
    for j in range(n_bits):
        diag *= np.where(bits[j] == 1, np.exp(-a * (4.0 ** j)), 1.0)
    for j in range(n_bits):
        for l in range(j + 1, n_bits):
            both = (bits[j] == 1) & (bits[l] == 1)
            diag *= np.where(both, np.exp(-2 * a * (2.0 ** (j + l))), 1.0)
    return diag


def test_product_form_matches_direct_exponentiation_exactly():
    for n_bits in (3, 5, 8):
        for a in (0.01, 0.37, 1.5):
            direct = _direct_diagonal(n_bits, a)
            product = _product_form_diagonal(n_bits, a)
            max_diff = np.max(np.abs(direct - product))
            assert max_diff < 1e-9, (n_bits, a, max_diff)


def _direct_diagonal_signed(n_bits, a):
    """Full fftfreq-ordered grid, including negative k (the upper half
    of the register in two's-complement convention) -- what a real
    momentum register actually represents, unlike the positive-k-only
    test above."""
    k = np.fft.fftfreq(2 ** n_bits, d=1.0 / (2 ** n_bits))
    return np.exp(-a * k.astype(float) ** 2)


def _naive_product_form_no_sign_correction(n_bits, a):
    """The ORIGINAL (AUDIT-2) construction, applied naively to the full
    signed grid by treating the stored register value m directly as k
    -- AUDIT-3 §3 showed this is wrong for the negative-k (upper) half."""
    return _product_form_diagonal(n_bits, a)  # m used directly, no sign handling


def _product_form_with_sign_correction(n_bits, a):
    """AUDIT-3 §3 fix: conditional two's-complement negation on the sign
    bit (MSB) FIRST, mapping the stored value m to |k|, THEN apply the
    unchanged product form on |k|'s bits. k^2 depends only on |k|, so
    this is exact, and the negation is O(n) Toffolis -- cheap, standard,
    and preserves alpha=1 (every factor remains a contraction, since
    |k| >= 0 always, unlike the naive version's amplifying factors)."""
    N = 2 ** n_bits
    m = np.arange(N)
    sign_bit = (m >> (n_bits - 1)) & 1
    m_abs = np.where(sign_bit == 1, N - m, m)
    m_abs = np.where(m == 0, 0, m_abs)

    bits = [(m_abs >> j) & 1 for j in range(n_bits)]
    diag = np.ones(N)
    for j in range(n_bits):
        diag *= np.where(bits[j] == 1, np.exp(-a * (4.0 ** j)), 1.0)
    for j in range(n_bits):
        for l in range(j + 1, n_bits):
            both = (bits[j] == 1) & (bits[l] == 1)
            diag *= np.where(both, np.exp(-2 * a * (2.0 ** (j + l))), 1.0)
    return diag


def test_naive_product_form_fails_on_negative_k_half_of_the_grid():
    """AUDIT-3 §3: confirms the failure mode exists before confirming
    the fix -- the naive extension (no sign handling) badly mismatches
    the true e^{-a*k^2} on the upper (negative-k, two's-complement)
    half of a real momentum register's grid."""
    n_bits, a = 6, 0.37
    direct = _direct_diagonal_signed(n_bits, a)
    naive = _naive_product_form_no_sign_correction(n_bits, a)
    max_diff = np.max(np.abs(direct - naive))
    assert max_diff > 0.1, max_diff  # badly wrong, not a rounding issue


def test_sign_corrected_product_form_matches_direct_exponentiation_on_full_signed_grid():
    """AUDIT-3 §3 fix, verified: conditional negation to |k| before the
    product form recovers exactness across the FULL fftfreq-ordered
    grid (both positive and negative k), not just the positive-k half
    the original (AUDIT-2) test covered."""
    for n_bits in (3, 5, 6, 8):
        for a in (0.01, 0.37, 1.5):
            direct = _direct_diagonal_signed(n_bits, a)
            fixed = _product_form_with_sign_correction(n_bits, a)
            max_diff = np.max(np.abs(direct - fixed))
            assert max_diff < 1e-9, (n_bits, a, max_diff)


def test_product_form_gate_count_matches_claimed_on_squared_scaling():
    """n singly-controlled + n(n-1)/2 doubly-controlled rotations per
    spatial direction -- confirms the O(n^2) claim used in the D4/G3
    gate-count accounting."""
    for n_bits in (4, 19):
        n_single = n_bits
        n_double = n_bits * (n_bits - 1) // 2
        total = n_single + n_double
        # O(n^2): total should be within a small constant of n_bits^2/2
        assert total <= n_bits ** 2, (n_bits, total)
        assert total >= n_bits, (n_bits, total)
