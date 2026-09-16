"""
D3 (constitution v2 §2, "encoding is free"): numerically verify the
quantics/TT-rank claim -- a sampled sine or cosine has exact TT-rank 2,
and u* (constant + product of two such 1D functions, interleaved bit
ordering) has chi(u*) <= 5, independent of N and Re.

`tt_ranks` is a standard sequential-SVD TT decomposition (TT-SVD /
Vidal/Oseledets algorithm): reshape a length-2^n vector into an n-site
tensor train of physical dimension 2 per site ("quantics"), and return
the bond dimension at each cut for a given truncation tolerance on the
relative singular-value mass. This is the classical instrument WS2 will
build a full FV solver in; here it only measures rank.
"""
import numpy as np


def tt_ranks(vec: np.ndarray, tol: float = 1e-10) -> list:
    """
    Sequential-SVD TT decomposition of a length-2^n vector into n sites
    of physical dimension 2. Returns the list of bond dimensions between
    consecutive sites (length n-1), truncated at singular values whose
    cumulative squared mass from the tail exceeds `tol` (relative to the
    total norm).
    """
    n = int(round(np.log2(len(vec))))
    assert 2 ** n == len(vec), "length must be a power of 2"

    ranks = []
    # C is the remaining "unfactored" block, initially the full vector
    # reshaped as (1, 2^n)
    C = vec.reshape(1, -1)
    r_left = 1
    total_norm = np.linalg.norm(vec)

    for site in range(n - 1):
        # reshape (r_left, 2^(n-site)) -> (r_left*2, 2^(n-site-1))
        remaining = C.shape[1]
        C = C.reshape(r_left * 2, remaining // 2)
        U, S, Vt = np.linalg.svd(C, full_matrices=False)

        r_new = _truncate_rank(S, tol, total_norm)
        ranks.append(r_new)

        U = U[:, :r_new]
        S = S[:r_new]
        Vt = Vt[:r_new, :]

        C = (S[:, None] * Vt)
        r_left = r_new

    return ranks


def _truncate_rank(S: np.ndarray, tol: float, total_norm: float) -> int:
    """Smallest r such that the discarded singular values contribute a
    relative Frobenius-norm error <= tol."""
    if len(S) == 0:
        return 1
    sq = S ** 2
    tail = np.sqrt(np.cumsum(sq[::-1]))[::-1]  # tail[k] = norm of S[k:]
    rel_tail = tail / total_norm
    # smallest r with rel_tail[r] <= tol (discarding S[r:] is within tol)
    r = np.argmax(rel_tail <= tol) if np.any(rel_tail <= tol) else len(S)
    return max(1, r)


def interleave_bits(i: np.ndarray, j: np.ndarray, n: int) -> np.ndarray:
    """
    Interleave the n-bit binary representations of i and j (each in
    [0, 2^n)) into a single 2n-bit index, bit by bit from the most
    significant: i_{n-1} j_{n-1} i_{n-2} j_{n-2} ... i_0 j_0.
    This is the standard quantics interleaved (row-major-in-scale)
    ordering that keeps a separable f(x)*g(y) field low-rank.
    """
    idx = np.zeros_like(i, dtype=np.int64)
    for b in range(n):
        bit_i = (i >> (n - 1 - b)) & 1
        bit_j = (j >> (n - 1 - b)) & 1
        idx = (idx << 1) | bit_i
        idx = (idx << 1) | bit_j
    return idx


def quantics_reshape_2d(field: np.ndarray) -> np.ndarray:
    """
    field: (N, N) array with N = 2^n, field[i, j] = value at grid point
    (i, j). Returns the length-4^n 1D array in interleaved-bit quantics
    order, so that a separable field f(x_i)*g(y_j) has TT-rank <=
    rank(f-as-quantics) * rank(g-as-quantics).
    """
    N = field.shape[0]
    n = int(round(np.log2(N)))
    assert 2 ** n == N and field.shape == (N, N)

    i_idx, j_idx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    flat_pos = interleave_bits(i_idx.ravel(), j_idx.ravel(), n)

    out = np.empty(N * N)
    out[flat_pos] = field.ravel()
    return out
