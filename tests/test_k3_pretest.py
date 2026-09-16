"""
K3 pre-test (constitution v2 §6 kill criterion): "if K3 (prescribed
spectral slope) requires manufactured forcing whose own tensor rank
exceeds the solution's, the knob is measuring the forcing rather than
the flow. Test at n=10 before building it out; cut K3 if it fails."

This is a DECISION GATE (Rule 4): if it fails, K3 is cut from the plan,
not worked around. We do not proceed to build a full K3 MMS/solver
integration unless this passes.

Numerical proxy (not full symbolic MMS -- a broadband multi-mode field's
symbolic forcing derivation is a large symbolic-algebra undertaking not
justified before knowing whether K3 survives at all): construct a
broadband 1D field with a prescribed spectral slope, measure its TT
rank, then measure the TT rank of its nonlinear self-advection term
u*du/dx (the dominant source of rank growth in the true MMS forcing,
since diffusion and pressure-gradient terms do not raise rank beyond
the field's own). If the nonlinear term's rank exceeds the field's rank,
K3 fails its own kill criterion.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from tgv.quantics import tt_ranks


def broadband_field(N, n_modes, alpha, rng, k_min=2, k_max=None):
    """Sum of n_modes random-phase Fourier modes with amplitude ~ k^-alpha
    (a prescribed spectral slope), sampled on N points over [0, 2*pi)."""
    if k_max is None:
        k_max = k_min + n_modes - 1
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    field = np.zeros(N)
    ks = np.arange(k_min, k_max + 1)
    phases = rng.uniform(0, 2 * np.pi, size=len(ks))
    for k, phase in zip(ks, phases):
        amp = k ** (-alpha)
        field += amp * np.sin(k * x + phase)
    return field, x


def test_k3_pretest_result_broadband_nonlinear_term_rank_exceeds_field_rank():
    """
    RESULT (recorded, not a bug): at n=10, N=1024, a broadband field
    with 5 random-phase modes and spectral slope alpha=1.5 has
    chi_field=6, but its nonlinear self-advection term u*du/dx has
    chi_nonlinear=8 > chi_field. The kill criterion in constitution v2
    §6 ("if K3 requires manufactured forcing whose own tensor rank
    exceeds the solution's, the knob is measuring the forcing rather
    than the flow") is TRIGGERED.

    Decision (Rule 4: never loosen a failed gate's threshold): **K3 is
    cut from the plan.** This test asserts the triggering finding
    itself, so it stays green and documents the negative result (Rule 3
    -- the graveyard is part of the deliverable) rather than blocking
    the suite on a gate that was designed to be checkable and, here,
    fails as designed.
    """
    N = 2 ** 10  # n = 10 bits, per the kill criterion's own test point
    rng = np.random.default_rng(0)

    results = []
    for n_modes, alpha in [(5, 1.5), (10, 1.5), (20, 1.5)]:
        field, x = broadband_field(N, n_modes=n_modes, alpha=alpha, rng=rng)
        dx = x[1] - x[0]

        chi_field = int(max(tt_ranks(field, tol=1e-6)))

        dfdx = (np.roll(field, -1) - np.roll(field, 1)) / (2 * dx)
        nonlinear_term = field * dfdx
        chi_nonlinear = int(max(tt_ranks(nonlinear_term, tol=1e-6)))

        results.append((n_modes, alpha, chi_field, chi_nonlinear))
        print(f"n_modes={n_modes} alpha={alpha}: "
              f"chi_field={chi_field}, chi_nonlinear={chi_nonlinear}",
              file=sys.stderr)

    # at least the smallest, most favorable case must already violate
    # the criterion for the cut decision to be justified
    n_modes, alpha, chi_field, chi_nonlinear = results[0]
    assert chi_nonlinear > chi_field, (
        "K3 pretest result changed -- re-examine the cut decision in "
        "STATUS.md before reinstating K3."
    )
