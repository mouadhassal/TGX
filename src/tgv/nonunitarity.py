"""
D4 (constitution v2 §2, "non-unitarity is untested, and the convection
term is why"): the information-theoretic floor on post-selection /
dilation cost is set by the norm decay ratio ||u(T)||/||u(0)||.

IMPORTANT (flagged as the weakest link in the constitution, §14): this
module computes a FLOOR, not an achieved algorithmic cost. A concrete
block-encoding's subnormalization can exceed this floor for reasons
unrelated to the solution's decay (WS4/A4 must compute an achieved cost
for at least one concrete algorithm to close the gap). Never report this
number as "the" cost.
"""
import numpy as np

from .config import TGVParams
from .analytic import mean_kinetic_energy


def norm_ratio_floor(p: TGVParams, T) -> float:
    """
    ||u(T)|| / ||u(0)|| = sqrt(E(T) / E(0)), the minimum possible
    post-selection / dilation penalty (its reciprocal) for representing
    this trajectory unitarily. Returns the ratio (in (0, 1] typically for
    a decaying flow).
    """
    E0 = mean_kinetic_energy(p, 0.0)
    ET = mean_kinetic_energy(p, T)
    return float(np.sqrt(ET / E0))


def dilation_penalty_floor(p: TGVParams, T) -> float:
    """Reciprocal of norm_ratio_floor: the minimum blow-up factor a
    dilation/LCU/post-selection scheme must pay, in the T -> infinity
    limit for a fixed Uc (worst case over T)."""
    E0 = mean_kinetic_energy(p, 0.0)
    E_inf = 0.5 * (p.Uc ** 2 + p.Vc ** 2)
    if E_inf <= 0:
        return float("inf")
    return float(np.sqrt(E0 / E_inf))


def one_ancilla_dilation_success_probability(p: TGVParams, T) -> float:
    """
    AUDIT.md D4 closure, LANGUAGE NARROWED per AUDIT-2 §2: this closes
    SUBNORMALIZATION exactly, for a generator diagonal in a known
    efficient basis (ours, via D2's QFT diagonalization) -- not the
    general case, and NOT gate count, which is a separate quantity (see
    below). Say "achieved alpha=1, exactly equal to the floor, with an
    explicit O(n^2) circuit" -- not "the D4 gap is closed," which reads
    as covering both.

    Construction: (1) QFT into the momentum register, where the
    generator is diagonal (D2); (2) one ancilla, with a controlled
    rotation R_y(2*arccos(|amplitude decay at that mode|)) conditioned
    on the momentum register, encoding each Fourier mode's own decay
    exp(-nu*k^2*T) as an ancilla-|0> amplitude; (3) post-select the
    ancilla on |0>. Every diagonal entry has magnitude <=1 (the
    generator is dissipative), so the block-encoding's subnormalization
    is alpha=1 EXACTLY -- not merely bounded -- and the post-selection
    success probability equals ||u(T)||^2/||u(0)||^2 exactly.

    GATE COST IS NOT FREE, BUT IS AVOIDABLE VIA A KNOWN TRICK (AUDIT-2
    §2, numerically verified in test_d4_product_form_dilation.py): the
    naive per-mode angle arccos(exp(-nu*k^2*T)) computed by general
    quantum arithmetic (square, scale, exponentiate, arccos) would cost
    real Toffolis and dominate everything else. Avoided because the
    exponent is a quadratic form in the qubit bits: with k=sum_j 2^j*b_j,
    e^{-a*k^2} factorizes into n singly-controlled and n(n-1)/2
    doubly-controlled R_y rotations with CLASSICALLY PRECOMPUTED angles
    -- O(n^2) controlled rotations per spatial direction, no runtime
    quantum arithmetic. Verified numerically to match direct
    exponentiation to <1e-9 for n up to 8.

    THREE CAVEATS, stated rather than buried (AUDIT-2 §2):
    - Subnormalization (this function) and gate count (O(n^2), above)
      are SEPARATE quantities; do not conflate "closed" claims.
    - Angle precision folds into eps: finite-precision rotations perturb
      the diagonal, and that error must enter the eps budget used
      elsewhere in this codebase (e.g. our_algorithm_query_complexity's
      eps), not sit outside it -- NOT separately accounted for here.
    - Amplitude amplification (the 1.22 = 1/sqrt(2/3) figure) assumes
      the initial state can be reflected about, i.e. re-prepared. If
      state prep dominates total cost (as it does in this codebase's
      G3 accounting, where readout/shots dominate the reported gate
      totals -- see quantum_resources.py), the 1/sqrt(p) factor
      multiplies the WHOLE prep+evolve block, not just the evolution
      step. This codebase's G3 numbers do NOT apply this correction;
      treat them as not yet including amplitude-amplification overhead.
    - Sign convention: RESOLVED (AUDIT-3 §3). The naive product-form
      construction, applied directly to a two's-complement-ordered
      register (as `np.fft.fftfreq` returns), is WRONG for the negative-
      k half: k^2 = m^2 - 2^(n+1)*m + 2^(2n) for the stored value m
      representing k=m-N, and the resulting middle-term factors have
      POSITIVE exponents (amplifications, not contractions), breaking
      alpha=1. Verified numerically
      (test_naive_product_form_fails_on_negative_k_half_of_the_grid).
      FIX, also verified: a conditional two's-complement negation on the
      sign bit (MSB) FIRST, mapping the register to |k|, THEN the
      unchanged product form on |k|'s bits -- exact because k^2 depends
      only on |k|, and O(n) Toffolis, cheap and standard. Verified to
      match direct exponentiation on the FULL signed grid (not just
      positive k) to <1e-9
      (test_sign_corrected_product_form_matches_direct_exponentiation_on_full_signed_grid).

    Returns the RAW post-selection success probability (>= 2/3 at the
    challenge parameters, per norm_ratio_floor). This is the raw
    probability; converting to an expected repetition count is 1/p
    (~1.5x at p=2/3), while boosting to near-certainty via amplitude
    amplification instead costs ~1/sqrt(p) (~1.22x at p=2/3, matching
    dilation_penalty_floor, subject to the re-preparation caveat above).
    State which convention is meant wherever this number is quoted.
    """
    ratio = norm_ratio_floor(p, T)
    return float(ratio ** 2)


def g5_dilation_penalty_advantage_window(p: TGVParams, T, c_reference: float) -> dict:
    """
    G5 (constitution v2 §11), PARAMETERIZED per AUDIT-2 §6 rather than
    left as a bare "undetermined" abstention: with D4's floor achieved
    EXACTLY for our own algorithm (one_ancilla_dilation_success_
    probability, c=1 by construction), the advantage window can be
    stated conditionally on the one genuinely unknown quantity -- a
    reference algorithm's own achieved-subnormalization-to-floor ratio
    c_reference (c=1 means it also achieves its floor exactly; c>1
    means its block-encoding pays more than the information-theoretic
    minimum, e.g. from a less efficient construction).

    The SAME floor applies to any algorithm solving this problem (it is
    a property of the physical trajectory -- E(T)/E(0) -- not of the
    algorithm), so a reference with slack c pays
    c * dilation_penalty_floor(p, T) while we pay exactly
    dilation_penalty_floor(p, T) (c=1). The window on THIS axis is open
    (we have a dilation-penalty advantage) whenever c_reference > 1.

    CAVEAT, stated rather than buried: this covers the dilation-penalty
    axis ONLY. It does NOT include the separate gate-count/query-
    complexity axis (see quantum_resources.py's G3 accounting, which is
    a different quantity, evaluated assuming c=1 implicitly and not
    parameterized here). A full window would combine both axes; this
    function converts one previously-abstained axis into a stated
    result, not the whole G5 question.
    """
    floor = dilation_penalty_floor(p, T)
    ours = floor
    reference = c_reference * floor
    return {
        "our_penalty": float(ours),
        "reference_penalty": float(reference),
        "window_open": bool(reference > ours),
        "advantage_factor": float(reference / ours) if ours > 0 else float("inf"),
        "caveat": "dilation-penalty axis only -- does not include the separate "
                  "gate-count/query-complexity axis (quantum_resources.py's G3 "
                  "accounting, evaluated at c=1 implicitly there)",
    }
