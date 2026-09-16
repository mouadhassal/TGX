"""
AUDIT-2 §5 resolution (option 1: resolve N3, confirm the headline
configuration achieves 2nd order): having confirmed H1 (dt~dx was not
aggressive enough to suppress temporal error relative to the O(dx^2)
spatial term, degrading the unforced case's apparent order --
test_n3_error_normalization.py), this test checks whether the ACTUAL
§7.1 headline runs (results/chi_of_re.json, produced by
scripts/measure_chi_of_re.py's cfl=0.3, dt=min(advective, diffusive)
formula) were affected.

They were not, by construction: the pre-registered N(Re) rule (Q6,
cell-Reynolds Re_Delta<=2) couples nu and dx precisely such that the
DIFFUSIVE constraint (dt ~ dx^2/(4*nu)) binds at all three headline
configurations (Re=10/N=32, Re=100/N=64, Re=1000/N=512) -- i.e. the
solver's own CFL logic was already selecting dt~dx^2 scaling there,
not the dt~dx (advective) scaling that caused H1's contamination in a
separately-chosen, coarser N range. This closes the loop: G2's own
gate (MMS-forced, order>=1.8) was never in question; the previously-
open N3 finding is now understood (H1) AND shown not to compromise the
specific numbers already reported in chi_of_re.json / writeup_draft.md
section 1.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tgv.config import TGVParams, N_of_Re


def test_headline_configurations_are_diffusive_bound_not_advective_bound():
    cfl = 0.3
    for Re in (10, 100, 1000):
        p = TGVParams(Re=Re)
        N = N_of_Re(Re)
        dx = p.domain_length / N
        dt_adv = cfl * dx / abs(p.Uc)
        dt_diff = cfl * dx ** 2 / (4 * p.nu)

        assert dt_diff < dt_adv, (
            f"Re={Re}, N={N}: diffusive dt ({dt_diff:.3e}) is NOT smaller than "
            f"advective dt ({dt_adv:.3e}) -- this headline point is advective-bound "
            f"(dt~dx, not dt~dx^2) and may carry H1's temporal-error contamination; "
            f"re-verify its order specifically before trusting it."
        )
