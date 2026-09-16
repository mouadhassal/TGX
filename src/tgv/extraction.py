"""
D6 (constitution v2 §2, "data extraction is the cheapest case that
exists"): both metrics the challenge names -- kinetic energy and L2
error against a KNOWN analytic solution -- are global quadratic
functionals, extractable to additive error eps in O(1/eps^2) samples
with NO dependence on N (dimension / grid resolution). K6 (a local probe
observable, e.g. one specific point's velocity or an integrated surface
force) is the knob that breaks this: extracting O(N) independent local
outputs to fixed per-output accuracy costs O(N) samples, not O(1).

This module verifies the N-independence (global) vs N-dependence (local,
under K6) empirically via a Monte Carlo sampling estimator, standing in
for the amplitude-estimation / swap-test sample complexity argument
(same Hoeffding/Chernoff scaling, [DERIVED]).
"""
import numpy as np

from .config import TGVParams
from .analytic import velocity, mean_kinetic_energy


def estimate_mean_ke_by_sampling(p: TGVParams, t: float, n_samples: int,
                                  rng: np.random.Generator) -> float:
    """
    Monte Carlo estimate of mean KE = <(u^2+v^2)/2> by evaluating the
    exact velocity field at random points in the periodic domain and
    averaging. This is the classical stand-in for amplitude estimation
    on an amplitude-encoded state: cost depends only on the desired
    accuracy epsilon, never on N (there IS no grid here at all).
    """
    x = rng.uniform(0, 2 * np.pi, size=n_samples)
    y = rng.uniform(0, 2 * np.pi, size=n_samples)
    u, v = velocity(p, x, y, t)
    ke_samples = 0.5 * (u ** 2 + v ** 2)
    return float(np.mean(ke_samples))


def samples_needed_for_global_functional(p: TGVParams, t: float, eps: float,
                                          rng: np.random.Generator,
                                          n_trials: int = 200) -> int:
    """
    Smallest n_samples (searched over a fixed grid of candidates) such
    that the empirical std error of the KE estimator is <= eps, i.e.
    O(1/eps^2) and, critically, independent of any grid resolution N.
    """
    for n in (10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000):
        estimates = [
            estimate_mean_ke_by_sampling(p, t, n, np.random.default_rng(seed))
            for seed in range(n_trials)
        ]
        std = float(np.std(estimates))
        if std <= eps:
            return n
    return -1  # did not converge in the searched range


def local_observable_samples_needed(n_outputs: int, eps: float) -> int:
    """
    K6 knob: to report n_outputs INDEPENDENT local observables (e.g. a
    probe velocity at n_outputs distinct points, or n_outputs components
    of a surface-force distribution) each to accuracy eps, a method that
    extracts one observable per measurement round needs n_outputs times
    the single-observable sample budget -- O(n_outputs / eps^2), unlike
    the O(1/eps^2) global-functional case. [DERIVED, standard
    concentration-inequality argument -- no field-specific structure to
    exploit once outputs are independent.]
    """
    per_observable = int(np.ceil(1.0 / eps ** 2))
    return n_outputs * per_observable
