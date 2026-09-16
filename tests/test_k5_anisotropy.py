"""
K5 knob (constitution v2 §3), grid anisotropy (Nx != Ny), NOT
viscosity anisotropy (AUDIT.md B1 -- an earlier version of this module
used nu_x != nu_y, which changes the PDE being solved and invalidates
the exact-solution reference; that was wrong and has been replaced).
nu stays a single isotropic value throughout, so the exact TGV solution
remains an exact reference at every anisotropy ratio.

CORRECTED CLAIM (AUDIT-2 §4): K5 does NOT, and structurally CANNOT,
break D5. The TGV occupies a single mode; the operator restricted to a
one-mode occupied subspace is a 1x1 block, whose condition number is
exactly 1 by definition, for every grid and every aspect ratio. The
tests below still measure something real and worth reporting (a
1.0000->1.0008 near-flat number, and a genuine, dramatic worst-case
kappa explosion), but NEITHER is evidence about D5's occupied-subspace
claim -- the occupied-kappa metric here is a knob that cannot move, not
a metric confirmed robust under stress. That correct, D5-breaking
knob is K3' (test_k3_prime.py::test_k3_prime_occupied_subspace_kappa_
grows_with_mode_count), which uses m incommensurate modes to make the
occupied subspace genuinely m-dimensional and its kappa tunable.

K5's own genuine, honest contribution: a real WALL-CLOCK cost from
grid anisotropy (the explicit diffusion-stability dt shrinks with
aspect ratio), demonstrated end-to-end in
test_k5_solver_grounded.py -- not a conditioning result.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from tgv.conditioning import (
    anisotropic_kappa,
    occupied_subspace_kappa_anisotropic,
)


def test_k5_isotropic_corner_matches_d5_kappa_one():
    Nx = 64
    Uc, Vc, nu, L = 1.0, 0.0, 2 * np.pi / 100, 2 * np.pi
    kappa = occupied_subspace_kappa_anisotropic(Nx, Nx, Uc, Vc, nu, L)
    assert np.isclose(kappa, 1.0, atol=1e-9), kappa


def test_k5_occupied_subspace_kappa_cannot_move_by_construction():
    """NOT evidence for D5 (see module docstring): a 1x1 block's
    condition number is 1 identically. This test documents that the
    tiny (~0.08%) movement seen is finite-grid discretization noise
    from comparing the same continuum mode on two different grids, not
    a tunable signal -- confirming the audit's point that this knob
    structurally cannot discriminate, rather than framing the flatness
    as a positive result."""
    Nx = 64
    Uc, Vc, nu, L = 1.0, 0.0, 2 * np.pi / 100, 2 * np.pi
    kappas = []
    for ratio in (1, 2, 5, 10):
        Ny = Nx * ratio
        kappas.append(occupied_subspace_kappa_anisotropic(Nx, Ny, Uc, Vc, nu, L))

    assert np.isclose(kappas[0], 1.0, atol=1e-9), kappas
    assert all(k < 1.01 for k in kappas), kappas  # never moves more than ~1% -- a null result


def test_k5_worst_case_kappa_explodes_this_is_a_real_cost_signal_not_a_d5_result():
    """The worst-case (operator-spectrum-tracking) kappa genuinely
    grows dramatically with grid aspect ratio -- a real, measurable
    fact about the discrete operator's spectrum. But paired against the
    occupied-subspace number above, this pairing is NOT a D5
    discriminator (see module docstring) -- it is one real signal
    (worst-case kappa growth) next to one structurally-null one
    (occupied kappa, which cannot move for a single-mode solution)."""
    Nx = 64
    Uc, Vc, nu, L = 1.0, 0.0, 2 * np.pi / 100, 2 * np.pi

    ratios = (1, 2, 5, 10)
    worst = [anisotropic_kappa(Nx, Nx * r, Uc, Vc, nu, L) for r in ratios]

    assert worst == sorted(worst), worst              # grows monotonically
    assert worst[-1] > 40 * worst[0], worst           # explodes (~50x measured)
