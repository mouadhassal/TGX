"""
D4: verify the norm-ratio floor derivation and its sharpest consequence --
that Uc = V0 = 1 bounds the penalty at ~1.22 for ALL Re, T, while the K4
knob (Uc = 0) makes it unbounded as T -> infinity.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import math
from tgv.config import TGVParams
from tgv.nonunitarity import norm_ratio_floor, dilation_penalty_floor


def test_d4_norm_ratio_floor_bounded_at_challenge_parameters():
    """E(0)=0.75, E(inf)=0.5 for Uc=1,Vc=0,V0=1 -> ratio >= sqrt(2/3)~=0.816
    -> penalty <= sqrt(3/2) ~= 1.2247, for EVERY Re and EVERY T."""
    for Re in (10, 100, 1e4, 1e6):
        p = TGVParams(Re=Re)
        for T in (0.1, 1, 10, 1e6):
            ratio = norm_ratio_floor(p, T)
            assert ratio >= math.sqrt(2.0 / 3.0) - 1e-9, (Re, T, ratio)
        penalty = dilation_penalty_floor(p, T=1e9)
        assert penalty <= math.sqrt(1.5) + 1e-6, (Re, penalty)
        assert math.isclose(penalty, math.sqrt(1.5), rel_tol=1e-6)


def test_d4_k4_uc_zero_makes_penalty_unbounded():
    """K4 knob: setting Uc=0 removes the surviving mean flow, so
    E(inf)=0 and the dilation penalty diverges as T -> infinity --
    exactly the non-unitarity difficulty the base parameters hide."""
    p = TGVParams(Re=100, Uc=0.0, Vc=0.0)
    penalty = dilation_penalty_floor(p, T=1e9)
    assert penalty == float("inf")

    # and it grows without bound as T increases (not just "large")
    ratios = [norm_ratio_floor(p, T) for T in (1, 10, 100, 1000)]
    assert ratios == sorted(ratios, reverse=True)  # monotonically decaying
    assert ratios[-1] < 1e-10  # effectively zero -> penalty -> infinity


def test_d4_penalty_matches_hand_derivation():
    """Cross-check against the closed-form E(t) directly, independent of
    mean_kinetic_energy's implementation."""
    p = TGVParams(Re=100)
    ratio = norm_ratio_floor(p, T=1e9)
    # E(inf) = 0.5*(1^2+0^2) = 0.5; E(0) = 0.5 + 0.25 = 0.75
    expected = math.sqrt(0.5 / 0.75)
    assert math.isclose(ratio, expected, rel_tol=1e-6)
