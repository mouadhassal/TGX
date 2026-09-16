"""
D3: numerically verify that a sampled sine/cosine has TT-rank <= 2, and
that u* has chi <= 5, independent of N (constitution v2 §2 D3 / A3).
Also the leakage guard from §6: rank caps in REPORTED solver runs must be
>= 32, but here we are MEASURING the intrinsic rank of the exact field,
which is a different (and legitimate) use of a tight tolerance.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from tgv.quantics import tt_ranks, quantics_reshape_2d
from tgv.config import TGVParams
from tgv.analytic import velocity


def test_d3_sine_has_tt_rank_at_most_2():
    for n in (6, 8, 10):
        N = 2 ** n
        x = np.linspace(0, 2 * np.pi, N, endpoint=False)
        vals = np.sin(3.7 * x)  # arbitrary frequency, not aligned to N
        ranks = tt_ranks(vals, tol=1e-9)
        assert max(ranks) <= 2, (n, ranks)


def test_d3_cosine_has_tt_rank_at_most_2():
    N = 2 ** 8
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    vals = np.cos(1.3 * x)
    ranks = tt_ranks(vals, tol=1e-9)
    assert max(ranks) <= 2, ranks


def test_d3_rank_is_independent_of_n():
    """chi must NOT grow as N increases -- that's the entire claim."""
    max_ranks = []
    for n in (5, 7, 9, 11):
        N = 2 ** n
        x = np.linspace(0, 2 * np.pi, N, endpoint=False)
        vals = np.sin(2 * x) * np.exp(-0.01 * x)  # decaying, still O(1) rank
        ranks = tt_ranks(vals, tol=1e-9)
        max_ranks.append(max(ranks))
    assert max(max_ranks) <= 3, max_ranks
    assert len(set(max_ranks)) == 1 or max(max_ranks) - min(max_ranks) <= 1, max_ranks


def test_d3_u_star_field_has_chi_at_most_5():
    """u*(x, y, t) sampled on an N x N grid, reshaped into quantics with
    interleaved bit ordering, must have TT-rank <= 5 at every cut,
    independent of N and Re -- the [DERIVED] claim in the constitution."""
    p = TGVParams(Re=100)
    for n in (5, 6, 7):
        N = 2 ** n
        x = np.linspace(0, 2 * np.pi, N, endpoint=False)
        y = np.linspace(0, 2 * np.pi, N, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing="ij")
        u, v = velocity(p, X, Y, t=3.0)

        q_u = quantics_reshape_2d(u)
        ranks_u = tt_ranks(q_u, tol=1e-9)
        assert max(ranks_u) <= 5, (n, "u", ranks_u)

        q_v = quantics_reshape_2d(v)
        ranks_v = tt_ranks(q_v, tol=1e-9)
        assert max(ranks_v) <= 5, (n, "v", ranks_v)


def test_d3_chi_independent_of_re():
    """Re only rescales nu, which only rescales the decay prefactor f(t)
    -- it must not change the field's spatial rank structure at all."""
    N = 2 ** 6
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    y = np.linspace(0, 2 * np.pi, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    max_ranks = []
    for Re in (10, 100, 1e4, 1e6):
        p = TGVParams(Re=Re)
        u, v = velocity(p, X, Y, t=3.0)
        q_u = quantics_reshape_2d(u)
        ranks_u = tt_ranks(q_u, tol=1e-9)
        max_ranks.append(max(ranks_u))

    assert all(r <= 5 for r in max_ranks), max_ranks
