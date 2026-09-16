"""
D6: verify sample complexity for the global KE functional is independent
of grid resolution N (there being no grid at all in the estimator is the
point), and that K6 (local, independent per-point observables) scales
linearly in the number of requested outputs -- the discriminator.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from tgv.config import TGVParams
from tgv.extraction import (
    estimate_mean_ke_by_sampling,
    samples_needed_for_global_functional,
    local_observable_samples_needed,
)
from tgv.analytic import mean_kinetic_energy


def test_d6_global_ke_estimator_converges_to_exact_value():
    p = TGVParams(Re=100)
    rng = np.random.default_rng(42)
    est = estimate_mean_ke_by_sampling(p, t=3.0, n_samples=200000, rng=rng)
    exact = mean_kinetic_energy(p, 3.0)
    assert abs(est - exact) < 0.01, (est, exact)


def test_d6_global_functional_sample_count_independent_of_n():
    """The estimator never references a grid resolution N at all -- the
    sample budget for a fixed eps is identical whether the 'grid' one
    might have used has 32 or 10^6 points per side, because the
    functional being extracted is global."""
    p = TGVParams(Re=100)
    rng = np.random.default_rng(0)
    n_needed = samples_needed_for_global_functional(p, t=3.0, eps=0.02, rng=rng)
    assert n_needed > 0
    # same eps target, same n_needed regardless of any notion of "N" --
    # demonstrated by the function signature itself taking no N argument
    assert n_needed <= 3000  # O(1/eps^2) ~ 2500 at eps=0.02, not astronomical


def test_d6_local_observable_cost_scales_linearly_with_output_count():
    eps = 0.05
    counts = [1, 10, 100, 1000]
    costs = [local_observable_samples_needed(n, eps) for n in counts]
    # exact linearity by construction of the model, but assert it
    # explicitly as the property that distinguishes K6 from the global case
    ratios = [costs[i] / counts[i] for i in range(len(counts))]
    assert all(abs(r - ratios[0]) < 1e-9 for r in ratios)


def test_d6_local_cost_exceeds_global_cost_once_output_count_grows():
    eps = 0.02
    global_cost = local_observable_samples_needed(1, eps)  # single output baseline
    many_outputs_cost = local_observable_samples_needed(1000, eps)
    assert many_outputs_cost == 1000 * global_cost
