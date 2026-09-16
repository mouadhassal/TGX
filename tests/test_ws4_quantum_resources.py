"""
WS4: confirms the structural comparisons in quantum_resources.py, and
the D4 "weakest link" (constitution v2 §14), REVISED per AUDIT.md B2
and the D4 closure it identified.

D4's floor (nonunitarity.py) is an INFORMATION-THEORETIC lower bound on
post-selection failure: ||u(T)||/||u(0)|| >= sqrt(2/3) at the challenge
parameters, for ANY algorithm. AUDIT.md showed this floor is ACHIEVED
EXACTLY by a concrete one-ancilla dilation for OUR algorithm
specifically (generator diagonal in a known efficient basis, D2) --
not the general case, where the gap remains genuinely open. Both halves
of that precise statement are tested below.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from tgv.quantum_resources import (
    general_fastforward_query_complexity,
    our_algorithm_query_complexity,
    our_algorithm_end_to_end_query_complexity,
    time_marching_query_complexity,
    time_marching_end_to_end_query_complexity,
    lbm_acoustic_scaling_dt,
    qlbm_reference_cost,
    carleman_qlsa_reference_cost,
)
from tgv.nonunitarity import norm_ratio_floor, dilation_penalty_floor, one_ancilla_dilation_success_probability
from tgv.config import TGVParams, N_of_Re


def test_our_algorithm_beats_general_fastforward_bound_in_t_scaling():
    """D2's structural claim, made precise: at large T, the general
    literature bound (sqrt(T)) grows without bound while our exact
    circulant-diagonalization cost (T_scaling='O(1)') does not -- this
    is the concrete content of 'our generator is exactly diagonalizable,
    the general algorithm does not assume that.'"""
    general = general_fastforward_query_complexity(T=1e12, eps=1e-6, mode="final_state")
    ours = our_algorithm_query_complexity(n_qubits=20, T=1e12, eps=1e-6)

    assert general["value_proxy"] > 1e5  # grows huge at large T
    assert ours["T_scaling"].startswith("O(1)")
    ours_small_T = our_algorithm_query_complexity(n_qubits=20, T=1.0, eps=1e-6)
    assert ours["total_gates_proxy"] == ours_small_T["total_gates_proxy"]


def test_qlbm_reference_cost_grows_with_re_per_jennings_et_al():
    """Sanity check the cited scaling law is monotonic and matches the
    re-verified combined exponent (1.936+0.375=2.311 empirical, 2D --
    AUDIT.md B3: the two factors multiply, they do NOT collapse to
    Re^(3D/8), a false simplification removed from the docstring)."""
    costs = [qlbm_reference_cost(Re, D=2, empirical=True)["scaling_proxy"]
             for Re in (10, 100, 1000, 1e4)]
    assert costs == sorted(costs)
    ratio = costs[1] / costs[0]  # Re: 10 -> 100
    expected_ratio = 10 ** (1.936 + 0.375)
    assert np.isclose(ratio, expected_ratio, rtol=1e-6)


def test_qlbm_analytic_bound_exponent_matches_re_verified_paper_at_d2():
    """AUDIT.md B3: at D=2, 3/4*(1+D/2) = 1.5, combined with
    q_M=Re^0.375 gives Re^1.875 total -- NOT Re^(3D/8)=Re^0.75, which
    was a false collapsed form in an earlier draft."""
    r = qlbm_reference_cost(Re=100, D=2, empirical=False)
    assert "3/4*(1+D/2)" in r["label"]
    # Re^1.875 at Re=100 vs Re^0.75 at Re=100: these must differ hugely,
    # confirming the code does NOT implement the false collapsed form
    wrong_collapsed_value = 100 ** 0.75 * (100 ** 0.375)
    assert not np.isclose(r["scaling_proxy"], wrong_collapsed_value, rtol=0.5)
    correct_value = (100 ** 1.5) * (100 ** 0.375)
    assert np.isclose(r["scaling_proxy"], correct_value, rtol=1e-6)


def test_carleman_qlsa_requires_lifted_kappa_not_naive_order_multiplication():
    """AUDIT.md B4: the function must NOT multiply a base kappa by
    carleman_order (that silently reintroduces the understatement bug).
    It only accepts an already-derived kappa_lifted and uses it as-is."""
    result = carleman_qlsa_reference_cost(kappa_lifted=100, eps=1e-3, carleman_order=2)
    assert "R<1" in result["caveat"] or "optimistic" in result["caveat"]
    assert "kappa_lifted" in result["caveat"]
    # doubling carleman_order with the SAME kappa_lifted must NOT change
    # the cost -- order is bookkeeping only, never multiplied in
    result_order4 = carleman_qlsa_reference_cost(kappa_lifted=100, eps=1e-3, carleman_order=4)
    assert result["query_count_proxy"] == result_order4["query_count_proxy"]


def test_g3_numeric_crossover_our_algorithm_at_k1_eq_1_vs_time_marching():
    """
    G3 (constitution v2 §11), REVISED AGAIN per AUDIT-2 §1 (the first
    revision anchored dt to an explicit-finite-difference diffusive
    stability limit, dt~dx^2/(4*nu) -- a CATEGORY ERROR, since QLBM's
    own timestep is not set by an FD von Neumann limit at all; and its
    "8.75e10 gates at 1 gate/step" summary line, without stating the
    n_shots=10000 readout multiplier alongside it, let a reader
    reconstruct an inconsistent dt and flag a false arithmetic bug --
    the CODE was consistent (8.75e6 steps x 10000 shots = 8.75e10), the
    REPORTING was not clear about which number included the shot
    factor).

    Fixed: dt now anchored to LBM's own textbook timestep rule --
    ACOUSTIC scaling, dt = dx/lambda, lambda~O(1) [FACT: standard LBM
    result, verified via WebSearch, distinct from the diffusive
    dt~dx^2 scaling that governs a different regime and was the wrong
    quantity for this reference]. And the two crossovers are reported
    SEPARATELY and labeled -- algorithmic-only (state prep alone, no
    readout) and end-to-end (including D6's global-functional shot
    model) -- so no reader has to reconstruct which number is which.

    SCOPE (also from AUDIT-2 §1): this argument bounds EXPLICIT
    time-marching references only. QLBM is explicit by construction, so
    this is the right scope for the actual comparison; it does NOT
    cover an implicit/global-in-time reference, which has no
    stability-derived dt ceiling and would need a different (per-step
    solve cost) argument not attempted here.
    """
    n_qubits = 19  # N_of_Re(1e6) = 2^19
    eps_state = 1e-3
    eps_readout = 1e-2
    k1_eps = 1.0  # K1 = 1, as G3 specifies

    p_re1e6 = TGVParams(Re=1e6)
    N = N_of_Re(1e6)
    dx = p_re1e6.domain_length / N
    # acoustic-scaling dt, lambda=1 (order the flow's own Uc=1) -- the
    # GENEROUS (large-dt, cheap-for-the-reference) choice
    dt_max = lbm_acoustic_scaling_dt(dx, lattice_velocity=1.0)

    # --- algorithmic-only crossover (state prep alone, no readout) ---
    ours_alg = our_algorithm_query_complexity(n_qubits=n_qubits, T=1.0, eps=eps_state, k1_eps=k1_eps)
    ours_alg_cost = ours_alg["total_gates_proxy"]
    T_star_algorithmic = ours_alg_cost * dt_max

    ref_alg_below = time_marching_query_complexity(T_star_algorithmic * 0.1, dt_max, gates_per_step=1.0)
    ref_alg_at = time_marching_query_complexity(T_star_algorithmic, dt_max, gates_per_step=1.0)
    ref_alg_above = time_marching_query_complexity(T_star_algorithmic * 10, dt_max, gates_per_step=1.0)
    assert ref_alg_below["total_gates_proxy"] < ours_alg_cost
    assert np.isclose(ref_alg_at["total_gates_proxy"], ours_alg_cost, rtol=1e-6)
    assert ref_alg_above["total_gates_proxy"] > ours_alg_cost

    # --- end-to-end crossover (state prep x readout shots, both sides) ---
    ours_e2e = our_algorithm_end_to_end_query_complexity(
        n_qubits=n_qubits, T=1.0, eps_state=eps_state, eps_readout=eps_readout, k1_eps=k1_eps)
    # readout dominates our own number here: n_shots=1/eps_readout^2 is
    # the large factor, common to both sides (AUDIT.md condition ii) --
    # stated explicitly, not left implicit
    n_shots = ours_e2e["n_shots_for_readout"]
    assert n_shots > ours_alg_cost  # readout, not the algorithm itself, dominates ours_e2e's total
    T_star_e2e = ours_e2e["state_prep_gates_per_shot"] * dt_max  # shots cancel: same T* as algorithmic

    ref_e2e_at = time_marching_end_to_end_query_complexity(
        T=T_star_e2e, dt=dt_max, eps_readout=eps_readout, gates_per_step=1.0)
    assert ref_e2e_at["n_shots_for_readout"] == n_shots
    assert np.isclose(ref_e2e_at["total_gates_proxy"], ours_e2e["total_gates_proxy"], rtol=1e-6)
    assert np.isclose(T_star_e2e, T_star_algorithmic, rtol=1e-6)  # shots cancel, as claimed

    # the actual point of G3: at the challenge's own T=10, are we ahead?
    assert T_star_algorithmic < 10.0, T_star_algorithmic
    challenge_ref_e2e = time_marching_end_to_end_query_complexity(
        T=10.0, dt=dt_max, eps_readout=eps_readout, gates_per_step=1.0)
    assert challenge_ref_e2e["total_gates_proxy"] > ours_e2e["total_gates_proxy"]

    print(f"dt (LBM acoustic scaling) = {dt_max:.3e}", file=sys.stderr)
    print(f"T* (algorithmic-only, no readout) = {T_star_algorithmic:.3e}", file=sys.stderr)
    print(f"T* (end-to-end, with readout -- shots cancel) = {T_star_e2e:.3e}", file=sys.stderr)
    print(f"ours algorithmic-only = {ours_alg_cost:.3e} gates", file=sys.stderr)
    print(f"ours end-to-end = {ours_e2e['total_gates_proxy']:.3e} gates "
          f"({n_shots} shots x {ours_e2e['state_prep_gates_per_shot']:.3e} gates/shot)", file=sys.stderr)
    print(f"reference end-to-end @ T=10 = {challenge_ref_e2e['total_gates_proxy']:.3e} gates", file=sys.stderr)


def test_d4_subnormalization_closed_for_our_algorithm_gate_count_is_separate():
    """
    AUDIT.md D4 closure, language narrowed per AUDIT-2 §2: SUBNORMALIZATION
    is closed exactly for our algorithm (alpha=1, achieved = floor) via
    an explicit one-ancilla dilation with a numerically-verified O(n^2)
    classical-angle circuit (test_d4_product_form_dilation.py) -- gate
    count is a separate quantity from subnormalization, not bundled
    into a single "the D4 gap is closed" claim (see nonunitarity.py's
    caveats on angle precision and amplitude-amplification
    re-preparation, neither folded into this codebase's G3 numbers).
    """
    p = TGVParams(Re=100)
    T = 1e9  # worst case over T, per the floor's own definition

    floor_ratio = norm_ratio_floor(p, T)
    achieved_prob = one_ancilla_dilation_success_probability(p, T)

    # achieved success probability EQUALS the floor's square exactly --
    # not merely bounded by it
    assert np.isclose(achieved_prob, floor_ratio ** 2, rtol=1e-9)
    assert achieved_prob >= 2.0 / 3.0 - 1e-6  # matches D4's >=2/3 statement

    # the two conventions the audit flagged must be distinguishable:
    # raw success probability (~2/3) vs. amplitude-amplified repetition
    # factor (~1.22 = 1/sqrt(2/3))
    naive_repetitions = 1.0 / achieved_prob          # ~1.5
    amplified_repetitions = 1.0 / np.sqrt(achieved_prob)  # ~1.22
    assert np.isclose(amplified_repetitions, dilation_penalty_floor(p, T), rtol=1e-6)
    assert not np.isclose(naive_repetitions, amplified_repetitions, rtol=0.1)


def test_d4_gap_remains_open_for_the_general_non_diagonal_case():
    """The precise, narrowed statement: the gap closes for OUR
    algorithm (previous test), but for a generator NOT diagonal in an
    efficiently computable basis, the one-ancilla construction above
    does not apply, and no code path in this repository computes an
    achieved subnormalization for that general case -- it remains open,
    honestly, rather than assumed closed by analogy."""
    general = general_fastforward_query_complexity(T=1e9, eps=1e-6, mode="final_state")
    assert general["value_proxy"] > 1e4  # grows with T -- a query count, not a subnormalization
    # no function in quantum_resources.py or nonunitarity.py divides
    # this query count by a floor to produce a general "achieved/floor"
    # ratio -- that absence is the documented open gap for the general
    # (non-diagonalizable) case.


def test_g5_conditional_advantage_window_parameterized_not_undetermined():
    """
    G5 (AUDIT-2 section 6): converted from a bare "undetermined"
    abstention into a stated, conditional result. At c_reference=1 (a
    reference that also achieves its own floor exactly), there is NO
    advantage on the dilation-penalty axis (tie, as expected -- both
    achieve the same information-theoretic floor). At any c_reference>1
    (a reference whose block-encoding is less efficient than the floor
    demands), our algorithm has a stated, quantified advantage on that
    axis.
    """
    from tgv.nonunitarity import g5_dilation_penalty_advantage_window

    p = TGVParams(Re=100)
    T = 10.0

    tie = g5_dilation_penalty_advantage_window(p, T, c_reference=1.0)
    assert tie["window_open"] is False
    assert np.isclose(tie["advantage_factor"], 1.0)

    advantage = g5_dilation_penalty_advantage_window(p, T, c_reference=2.5)
    assert advantage["window_open"] is True
    assert np.isclose(advantage["advantage_factor"], 2.5)
    assert advantage["reference_penalty"] > advantage["our_penalty"]
