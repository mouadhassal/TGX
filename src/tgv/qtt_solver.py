"""
Item #2 (the deferred "brilliance upgrade"): a genuinely compressed
quantics/TT representation and evaluation of the TGV field, built directly
from D1-D3's own structural results rather than post-hoc SVD-compressing a
dense solve after the fact.

D3 (quantics.py) proved by SVD measurement that the exact field's quantics
rank is <=5. This module proves the same bound CONSTRUCTIVELY: it builds an
explicit rank-5 transfer-matrix realization of the interleaved-bit quantics
representation directly from the closed-form solution's own algebraic shape
(a constant plus sin(x)*cos(y), and a linear-phase trig function of a
binary-encoded integer has an exact rank-2 rotation-matrix TT by the
angle-addition formula: R(a)R(b) = R(a+b)). Field values and the mean
kinetic energy are then read out through O(n_bits) small matrix
contractions, at a cost and a memory footprint independent of N. No dense
(N, N) array is ever formed, at any N, including N=2^19 (Re=10^6), the
point every earlier draft of this report could only extrapolate to.

Time enters through exactly two scalars folded into boundary vectors: a
phase offset (from Uc*t/L, Vc*t/L) and a decay amplitude f(t) =
exp(-2*nu*t/L^2) -- D2's own fast-forward eigenvalue. The interior transfer
matrices never change with t. This is D2's "diagonalize once, multiply by
exp(t*lambda)" fast-forward, carried out in the quantics/TT basis instead
of the Fourier basis it was originally verified in.

Scope, stated precisely, per the constitution's Rule 1: this evaluates and
reads out the EXACT solution in compressed form. It is not a from-scratch
nonlinear-PDE solve carried out in TT arithmetic -- D1 already establishes
there is no nonlinear term to solve for on this trajectory, and building a
TT time-marching scheme to rediscover that by brute force would be slower,
more complex, and would not prove anything D1 and D2 had not already
proven. What this module adds is the missing constructive link between
that fact and an actual compressed solver artifact: something that runs,
at Re=10^6, and produces a real, checkable number.
"""
import numpy as np

from .config import TGVParams

_R = 5  # 4 oscillatory (cx*cy, cx*sy, sx*cy, sx*sy) + 1 constant


def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _site_weights(n_bits: int) -> list:
    """Weight of bit b (b=0 is MSB, matching quantics.interleave_bits)."""
    return [2 ** (n_bits - 1 - b) for b in range(n_bits)]


def build_field_qtt(p: TGVParams, t: float, n_bits: int) -> dict:
    """
    Build the exact rank-<=5 transfer-matrix quantics representation of
    (u, v) at time t on an N=2^n_bits per-axis grid of point samples
    x_i = i*dx, dx = domain_length/N -- without ever forming the (N, N)
    dense array. Cost and memory here are O(n_bits), not O(N^2).
    """
    assert n_bits >= 1
    N = 2 ** n_bits
    dx = p.domain_length / N
    ax = dx / p.L
    ay = dx / p.L
    bx = -p.Uc * t / p.L
    by = -p.Vc * t / p.L
    f = np.exp(-2.0 * p.nu * t / p.L ** 2)

    wx = _site_weights(n_bits)
    wy = _site_weights(n_bits)
    I2 = np.eye(2)

    init_osc = np.kron([np.cos(bx), np.sin(bx)], [np.cos(by), np.sin(by)])
    init = np.concatenate([init_osc, [1.0]])

    # interleaved sites: x-bit, y-bit, x-bit, y-bit, ... (2*n_bits total),
    # matching quantics.interleave_bits's bit order exactly.
    site_mats = np.zeros((2 * n_bits, 2, _R, _R))
    for b in range(n_bits):
        Rx1 = _rot(ax * wx[b])
        Ry1 = _rot(ay * wy[b])
        for bitval, Rx in ((0, I2), (1, Rx1)):
            site_mats[2 * b, bitval, :4, :4] = np.kron(Rx, I2)
            site_mats[2 * b, bitval, 4, 4] = 1.0
        for bitval, Ry in ((0, I2), (1, Ry1)):
            site_mats[2 * b + 1, bitval, :4, :4] = np.kron(I2, Ry)
            site_mats[2 * b + 1, bitval, 4, 4] = 1.0

    # read vectors, index order [cx*cy, cx*sy, sx*cy, sx*sy, const]
    read_u = np.array([0.0, 0.0, p.V0 * f, 0.0, p.Uc])
    read_v = np.array([0.0, -p.V0 * f, 0.0, 0.0, p.Vc])

    return {
        "site_mats": site_mats, "init": init,
        "read_u": read_u, "read_v": read_v, "n_bits": n_bits,
    }


def evaluate_at(qtt: dict, i: int, j: int):
    """Point value (u, v) at grid indices (i, j), via O(n_bits) matrix-vector
    products -- no dense array is consulted."""
    n_bits = qtt["n_bits"]
    state = qtt["init"].copy()
    for b in range(n_bits):
        bit_i = (i >> (n_bits - 1 - b)) & 1
        bit_j = (j >> (n_bits - 1 - b)) & 1
        state = qtt["site_mats"][2 * b, bit_i] @ state
        state = qtt["site_mats"][2 * b + 1, bit_j] @ state
    return float(qtt["read_u"] @ state), float(qtt["read_v"] @ state)


def _sum_of_squares(qtt: dict, read: np.ndarray) -> float:
    """sum over ALL (i, j) in [0,N)^2 of (read . state(i,j))^2, computed via
    the standard transfer-operator-squared trick (T = sum_bit M(bit) (x)
    M(bit)) so that the whole grid is summed in O(n_bits) steps on an
    R^2 x R^2 object, independent of N."""
    n_bits = qtt["n_bits"]
    state2 = np.kron(qtt["init"], qtt["init"])
    for site in range(2 * n_bits):
        M0 = qtt["site_mats"][site, 0]
        M1 = qtt["site_mats"][site, 1]
        T = np.kron(M0, M0) + np.kron(M1, M1)
        state2 = T @ state2
    return float(np.kron(read, read) @ state2)


def mean_kinetic_energy_qtt(p: TGVParams, t: float, n_bits: int) -> float:
    """Mean KE = (1/(2N^2)) * sum(u^2+v^2), read out entirely from the
    compressed representation -- N never appears as an array size, only as
    the scalar N^2 in the final division."""
    qtt = build_field_qtt(p, t, n_bits)
    N = 2 ** n_bits
    sum_u2 = _sum_of_squares(qtt, qtt["read_u"])
    sum_v2 = _sum_of_squares(qtt, qtt["read_v"])
    return (sum_u2 + sum_v2) / (2.0 * N * N)


def qtt_memory_bytes(qtt: dict) -> int:
    """Actual measured memory of the compressed representation: the site
    matrices plus the four boundary/read vectors. O(n_bits), independent
    of N -- a real number, not a claim."""
    return (qtt["site_mats"].nbytes + qtt["init"].nbytes
            + qtt["read_u"].nbytes + qtt["read_v"].nbytes)


def contract_dense(p: TGVParams, t: float, n_bits: int):
    """Verification-only: contract the QTT to the full (N, N) dense (u, v)
    arrays by brute-force point evaluation. Only ever called at small
    n_bits in tests -- this is the O(N^2) path the compressed solver
    exists to avoid, kept only so it can be checked against."""
    qtt = build_field_qtt(p, t, n_bits)
    N = 2 ** n_bits
    u = np.zeros((N, N))
    v = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            u[i, j], v[i, j] = evaluate_at(qtt, i, j)
    return u, v
