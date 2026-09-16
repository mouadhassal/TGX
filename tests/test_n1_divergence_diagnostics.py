"""
N1 (AUDIT.md): the compact-stencil `divergence()` diagnostic shares the
pressure projector's own blind spot (its Fourier symbol vanishes at
Nyquist, the same symbol the projector pins to zero) -- so
max_div ~ 1e-14 from that function alone does not prove the field is
genuinely checkerboard-free. `divergence_spectral()` uses the continuum
symbol instead, which IS sensitive to Nyquist/checkerboard content, and
is used here as an independent cross-check.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from tgv.fv_solver import divergence, divergence_spectral, run
from tgv.config import TGVParams
from tgv.analytic import velocity


def test_n1_compact_stencil_is_blind_to_pure_checkerboard_in_u():
    """A pure checkerboard pattern in u (period-2 in x, i.e. exactly the
    Nyquist mode) with v=0 has a non-zero TRUE divergence (du/dx is not
    globally zero), but the compact central-difference stencil's
    Fourier symbol vanishes exactly at Nyquist, so `divergence()` must
    report (numerically) zero for it -- demonstrating the blind spot
    concretely, not just asserting it exists."""
    N = 32
    dx = dy = 2 * np.pi / N
    i = np.arange(N)
    checkerboard_1d = (-1.0) ** i
    u = np.tile(checkerboard_1d[:, None], (1, N))  # varies only in x, period 2
    v = np.zeros_like(u)

    div_compact = divergence(u, v, dx, dy)
    div_spectral = divergence_spectral(u, v, dx, dy)

    assert np.max(np.abs(div_compact)) < 1e-10, np.max(np.abs(div_compact))  # blind
    assert np.max(np.abs(div_spectral)) > 1.0, np.max(np.abs(div_spectral))  # sees it


def test_n1_solver_output_has_no_hidden_checkerboard_on_the_smooth_tgv_case():
    """Cross-check the actual solver's output (base TGV, smooth,
    well-resolved) with BOTH diagnostics: both should agree the field is
    genuinely divergence-free, not just compact-stencil-blind."""
    N = 32
    p = TGVParams(Re=100)
    dx = p.domain_length / N
    xs = (np.arange(N) + 0.5) * dx
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    u0, v0 = velocity(p, X, Y, 0.0)

    dt = 0.2 * dx / abs(p.Uc)
    u, v, press, _ = run(u0, v0, dx, dx, p.nu, dt, n_steps=20)

    max_div_compact = np.max(np.abs(divergence(u, v, dx, dx)))
    max_div_spectral = np.max(np.abs(divergence_spectral(u, v, dx, dx)))

    assert max_div_compact < 1e-9, max_div_compact
    # one-sided diagnostic is a genuinely independent check (different
    # symbol, nonzero at Nyquist); on this smooth case it should be
    # small and, more rigorously, shrink under grid refinement at its
    # OWN 1st-order rate -- distinguishing genuine O(dx) truncation
    # (which shrinks) from hidden checkerboard energy (which would not
    # shrink cleanly at a fixed rate under refinement).
    assert max_div_spectral < 0.15, max_div_spectral  # O(dx) truncation floor at N=32, dx~0.2


def test_n1_one_sided_diagnostic_shrinks_at_first_order_under_refinement():
    """Distinguishes genuine truncation error from hidden checkerboard
    noise: if max_div_spectral were picking up real checkerboard energy
    rather than its own O(dx) truncation error, it would NOT shrink
    cleanly by ~2x per doubling of N."""
    Re = 100
    p = TGVParams(Re=Re)
    ratios = []
    prev = None
    for N in (32, 64, 128):
        dx = p.domain_length / N
        xs = (np.arange(N) + 0.5) * dx
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        u0, v0 = velocity(p, X, Y, 0.0)
        dt = 0.2 * dx / abs(p.Uc)
        u, v, press, _ = run(u0, v0, dx, dx, p.nu, dt, n_steps=20)
        m = np.max(np.abs(divergence_spectral(u, v, dx, dx)))
        if prev is not None:
            ratios.append(prev / m)
        prev = m

    # 1st order: each doubling of N should roughly halve->quarter max_div
    # (ratio ~1.5-2.5 is consistent with 1st order plus some 2nd-order
    # solver error mixed in; a ratio near 1 would indicate non-shrinking
    # noise, i.e. genuine checkerboard content)
    assert all(r > 1.3 for r in ratios), ratios
