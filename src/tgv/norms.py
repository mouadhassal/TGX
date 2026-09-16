"""
Error norms and conservation/symmetry diagnostics (WS0.3, constitution v2
§9.3-9.4). All norms are defined on CELL AVERAGES, never point samples
(Q3 [ASSUMPTION]).
"""
import numpy as np


def relative_l2_velocity(u_h, v_h, u_ref, v_ref):
    """
    Q3: relative L2 error on the velocity vector,
        eps2 = ||u_h - u_ref||_2 / ||u_ref||_2
    computed over a field of cell-average values (both arguments must
    already be cell averages on the same grid).
    """
    u_h = np.asarray(u_h, dtype=float)
    v_h = np.asarray(v_h, dtype=float)
    u_ref = np.asarray(u_ref, dtype=float)
    v_ref = np.asarray(v_ref, dtype=float)

    num = np.sqrt(np.sum((u_h - u_ref) ** 2 + (v_h - v_ref) ** 2))
    den = np.sqrt(np.sum(u_ref ** 2 + v_ref ** 2))
    return num / den


def relative_l2_fluctuation(u_h, v_h, u_ref, v_ref, Uc, Vc):
    """
    AUDIT.md N3: normalize by the FLUCTUATING (vortex) part's own norm,
    not the total field's -- `relative_l2_velocity` divides by
    ||u_ref|| including the non-decaying mean flow (Uc, Vc), which
    DILUTES the visible error as the vortex survives less at fixed T
    (e.g. at low Re, where e^{-2*nu*T} is tiny by T=10): the required
    "error scaling vs Re" curve then measures vortex survival as much
    as numerical quality. This function divides by the fluctuation's
    own norm instead, isolating genuine discretization quality. The
    numerator is unchanged (Uc, Vc cancel exactly there already, since
    they are additive constants -- this function only changes what the
    error is measured RELATIVE TO).
    """
    u_h = np.asarray(u_h, dtype=float)
    v_h = np.asarray(v_h, dtype=float)
    u_ref = np.asarray(u_ref, dtype=float)
    v_ref = np.asarray(v_ref, dtype=float)

    num = np.sqrt(np.sum((u_h - u_ref) ** 2 + (v_h - v_ref) ** 2))
    den = np.sqrt(np.sum((u_ref - Uc) ** 2 + (v_ref - Vc) ** 2))
    return num / den


def kinetic_energy_decay_error(E_h_t, E_exact_t):
    """Absolute error between a solver's reported mean KE trajectory and
    the exact E(t), evaluated pointwise in time (arrays aligned by caller)."""
    return np.abs(np.asarray(E_h_t, dtype=float) - np.asarray(E_exact_t, dtype=float))


def max_divergence(u, v, dx, dy):
    """
    max|div(u)| on a cell-centered field using a periodic central
    difference of face-normal velocities. Conservation check, run on
    every commit (Rule 5 / §9.4).
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    dudx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dx)
    dvdy = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2 * dy)
    return np.max(np.abs(dudx + dvdy))


def energy_drift(E_h_t, E_exact_t):
    """Signed drift of solver KE from exact KE at each reported time, and
    its max magnitude (the number to threshold in G-gates)."""
    E_h_t = np.asarray(E_h_t, dtype=float)
    E_exact_t = np.asarray(E_exact_t, dtype=float)
    drift = E_h_t - E_exact_t
    return drift, np.max(np.abs(drift))


def galilean_check(err_a, err_b, tol=1e-8):
    """
    §9.4: running with (Uc,Vc) = (0,0) vs (1,0) must give identical error
    up to translation, since the exact solution is a Galilean shift
    [DERIVED from D1]. err_a, err_b are matched error arrays (e.g. eps2(t)
    for the two runs, on grids related by the same shift). Returns
    (passed: bool, max_abs_diff: float).
    """
    err_a = np.asarray(err_a, dtype=float)
    err_b = np.asarray(err_b, dtype=float)
    diff = np.max(np.abs(err_a - err_b))
    return diff <= tol, diff
