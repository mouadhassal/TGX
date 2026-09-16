"""
"Brilliant upgrade" item #4: a concrete phase diagram for G5's advantage
window, using K3''s ALREADY-MEASURED chi(m) and kappa(m) data (no new
experiment invented) rather than the abstract c-parameterized statement
in g5_dilation_penalty_advantage_window.

Cost models, each tagged honestly:
  - Classical instrument (TT/quantics arithmetic): cost ~ chi^2
    [ASSUMPTION: standard two-index TT-contraction scaling; the
    constitution's own ablation table proposed chi^3 empirically but
    that ablation was never run in this codebase, so chi^2 -- the more
    conservative, more standard estimate -- is used here instead, and
    tagged as an assumption, not a measured law].
  - Quantum reference (Carleman+QLSA): cost ~ kappa
    [FACT: standard QLSA query complexity, O(kappa*polylog(1/eps)),
    already implemented in quantum_resources.carleman_qlsa_reference_cost].

Both cost drivers (chi, kappa) ARE real, measured data from
test_k3_prime.py, not fabricated. Absolute unit conversion between the
two cost models is NOT attempted (Rule 1: no invented constants) --
instead both are normalized to their own m=1 value and compared as
RELATIVE growth, which is a well-posed, honest question: "which cost
driver grows faster as this knob turns."
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    # measured, tolerance-converged (test_k3_prime.py)
    m_values = [1, 2, 4, 8]
    chi = [4, 8, 16, 24]
    kappa = [1.0, 9.0, 49.0, 361.0]

    chi2 = [c ** 2 for c in chi]
    chi2_norm = [c / chi2[0] for c in chi2]
    kappa_norm = [k / kappa[0] for k in kappa]

    print("G5 phase diagram from K3' (real, measured data -- test_k3_prime.py):")
    print()
    print(f"{'m':>3} | {'chi':>5} | {'chi^2 (norm.)':>14} | {'kappa (norm.)':>14} | {'which grows faster':>20}")
    print("-" * 72)
    for i, m in enumerate(m_values):
        faster = "kappa (quantum ref.)" if kappa_norm[i] > chi2_norm[i] else "chi^2 (classical)"
        print(f"{m:3d} | {chi[i]:5d} | {chi2_norm[i]:14.1f} | {kappa_norm[i]:14.1f} | {faster:>20}")

    print()
    print("HONEST FINDING, not the hoped-for direction:")
    print("kappa (the quantum-reference cost driver) grows FASTER than")
    print("chi^2 (the classical cost driver) at every step of this knob --")
    print("9x vs 4x at m=2, 49x vs 16x at m=4, 361x vs 36x at m=8. There is")
    print("no crossover in this direction across the tested range: on THIS")
    print("cost-model pairing, K3' makes the CLASSICAL instrument relatively")
    print("MORE favored as the knob turns, not less. Reported as measured,")
    print("per Rule 5 (never manufacture a window in the hoped-for direction).")
    print()
    print("This does not contradict G3 (our OWN quantum algorithm beats a")
    print("time-marching reference at K1=1): that crossover concerns a")
    print("different pair of methods (our fast-forwarding algorithm vs.")
    print("naive time-marching) and a different knob (K1, not K3'). The")
    print("result here is specifically about Carleman+QLSA's kappa-driven")
    print("cost vs. the classical TT instrument's chi-driven cost, under K3'.")


if __name__ == "__main__":
    main()
