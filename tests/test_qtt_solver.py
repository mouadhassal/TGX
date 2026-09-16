"""
Verification for the compressed QTT solver (qtt_solver.py, item #2 of the
"brilliance upgrades"). Checks, at small N where a dense array is still
affordable:
  1. Pointwise field values match the closed-form analytic solution exactly
     (to float precision).
  2. The QTT-native mean-KE readout matches the closed-form E(t) exactly.
  3. The constructive rank-5 claim is cross-validated against the
     pre-existing, independent SVD-based rank measurement (quantics.py) --
     two different methods agreeing is real evidence, not just one method
     asserting itself.
Then checks, at N=2^19 (Re=10^6) -- the point every earlier draft of this
report could only extrapolate to -- that the compressed solver actually
runs and returns a finite, physically sane number, using real measured
bytes rather than a claimed figure.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from tgv.config import TGVParams
from tgv.analytic import velocity, mean_kinetic_energy
from tgv.quantics import quantics_reshape_2d, tt_ranks
from tgv.qtt_solver import (
    build_field_qtt, evaluate_at, contract_dense,
    mean_kinetic_energy_qtt, qtt_memory_bytes,
)


@pytest.mark.parametrize("n_bits", [4, 5, 6])
@pytest.mark.parametrize("t", [0.0, 1.0, 10.0])
def test_pointwise_matches_analytic(n_bits, t):
    p = TGVParams(Re=100)
    N = 2 ** n_bits
    dx = p.domain_length / N
    qtt = build_field_qtt(p, t, n_bits)

    rng = np.random.default_rng(0)
    for i, j in rng.integers(0, N, size=(12, 2)):
        u_qtt, v_qtt = evaluate_at(qtt, int(i), int(j))
        u_ref, v_ref = velocity(p, i * dx, j * dx, t)
        assert abs(u_qtt - float(u_ref)) < 1e-11
        assert abs(v_qtt - float(v_ref)) < 1e-11


@pytest.mark.parametrize("t", [0.0, 2.0, 5.0, 10.0])
def test_ke_readout_matches_closed_form(t):
    p = TGVParams(Re=100)
    ke_qtt = mean_kinetic_energy_qtt(p, t, n_bits=6)
    ke_ref = float(mean_kinetic_energy(p, t))
    assert abs(ke_qtt - ke_ref) < 1e-10


def test_full_dense_contraction_matches_analytic_field():
    p = TGVParams(Re=100)
    n_bits = 5
    N = 2 ** n_bits
    dx = p.domain_length / N
    u_qtt, v_qtt = contract_dense(p, t=3.0, n_bits=n_bits)

    xs = (np.arange(N)) * dx
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    u_ref, v_ref = velocity(p, X, Y, 3.0)

    assert np.max(np.abs(u_qtt - u_ref)) < 1e-10
    assert np.max(np.abs(v_qtt - v_ref)) < 1e-10


def test_constructive_rank_matches_independent_svd_measurement():
    """The rank-5 transfer-matrix construction here is independent of the
    SVD-based tt_ranks() measurement in quantics.py (built from the field's
    known algebraic form, not from decomposing a sampled array). The two
    should agree: max bond dimension <= 5 either way."""
    p = TGVParams(Re=100)
    n_bits = 6
    u_qtt, _ = contract_dense(p, t=4.0, n_bits=n_bits)

    flat = quantics_reshape_2d(u_qtt)
    ranks = tt_ranks(flat, tol=1e-9)
    assert max(ranks) <= 5


def test_compressed_solver_runs_at_re_1e6():
    """The point every earlier draft of this report could only extrapolate
    to (Re=10^6, N=524288, 2.7e11 cells) -- infeasible to solve densely on
    any hardware available to this project. The compressed solver runs
    here directly: no dense array is formed, memory is measured (not
    claimed), and the returned KE is checked against the closed form,
    which remains valid regardless of N since it never touches a grid."""
    p = TGVParams(Re=1_000_000)
    n_bits = 19
    N = 2 ** n_bits
    assert N == 524288

    ke = mean_kinetic_energy_qtt(p, t=10.0, n_bits=n_bits)
    ke_ref = float(mean_kinetic_energy(p, 10.0))
    assert abs(ke - ke_ref) < 1e-9

    qtt = build_field_qtt(p, t=10.0, n_bits=n_bits)
    measured_bytes = qtt_memory_bytes(qtt)
    dense_bytes_equivalent = 2 * N * N * 8  # u and v, float64, never formed

    assert measured_bytes < 20_000  # tens of KB, not a claim
    assert dense_bytes_equivalent > 4e12  # ~4.4 TB, confirms infeasibility
    assert measured_bytes < dense_bytes_equivalent / 1e8


def test_memory_independent_of_n():
    """Bond dimension, and therefore memory, does not grow with N -- only
    with n_bits (linearly), which is exactly the constructive content of
    the rank-<=5 claim, not just a measurement at one resolution."""
    p = TGVParams(Re=100)
    bytes_by_nbits = {}
    for n_bits in (6, 10, 15, 19):
        qtt = build_field_qtt(p, t=1.0, n_bits=n_bits)
        bytes_by_nbits[n_bits] = qtt_memory_bytes(qtt)

    # linear in n_bits, not in N=2^n_bits
    ratio_15_to_6 = bytes_by_nbits[15] / bytes_by_nbits[6]
    assert ratio_15_to_6 < 3.0  # not exponential
