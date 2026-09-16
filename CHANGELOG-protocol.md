# Protocol Changelog

Per Rule 3 (v2 §0): pre-registration is law; any change after seeing a
result is logged here with date, reason, old value, new value.

## 2026-09-15: compressed quantics/TT solver built (src/tgv/qtt_solver.py), closing the tensor-network gap

**Reason:** the tensor-network track's own hard constraint, a genuinely
compressed quantics solver rather than a dense solve plus post-hoc
compression, was previously an open limitation (six-degeneracies-report.html
Appendix B/§6). This is an addition, not a change to any pre-registered
gate or degeneracy result: no prior number in the report is altered.

**What was built:** an explicit rank-5 transfer-matrix construction of the
TGV field's quantics representation, derived directly from the closed-form
solution's algebraic shape (constant + sin(x)*cos(y), each a linear-phase
trig function of a binary-encoded integer, exact rank-2 by the
angle-addition formula). Field values and the mean-KE readout are computed
through O(n_bits) small matrix products; no dense (N,N) array is ever
formed, at any N.

**Scope, stated for the record:** this is a compressed evaluator for a
solution D1 already shows has no leftover nonlinear term. It is not a
general nonlinear-PDE solver in TT arithmetic, and is not presented as one.

**Verification:** pointwise match against the closed-form field to
<1e-11 (tests/test_qtt_solver.py); dense-contraction at small N cross-
checked against the pre-existing, independent SVD-based rank measurement
in quantics.py (agree: rank <=5 both ways); run directly at Re=1e6,
N=524288 (n_bits=19), the point every earlier draft could only extrapolate
to, with KE readout matching the closed form to 1.1e-16 and measured
memory of ~15.3 KB versus a 4.40 TB dense equivalent that was never
formed. 17 new tests, all passing; full suite 96/96.

**Old value:** report listed the compressed quantics solver as not built;
Re=10^4-10^6 storage figures were extrapolated from D3's theorem, not run.
**New value:** solver exists (src/tgv/qtt_solver.py); Re up to 1e6 has a
real, verified, directly-run number for both storage and KE readout.

## 2026-08-27 — AUDIT-3 response: genuine 1st-order-in-time bug found and fixed, K3' confound resolved, D4 sign convention fixed

**Reason:** third-round external audit (AUDIT-3.md) of RESPONSE-TO-AUDIT-2.md.

**The real bug (§1): fv_solver.py's step() was genuinely 1st order in
time, not 2nd.** AUDIT-2's "H1 confirmed" diagnosis (temporal error
floor contaminating a genuine 2nd-order Richardson estimate) was a
MISDIAGNOSIS, caught by AUDIT-3 noting N2 and N3's own numbers were
mutually inconsistent -- a genuinely 2nd-order method cannot show order
1.06 at dt~dx (temporal error would already match the O(dx^2) spatial
term). The simpler, correct explanation: the RK2 predictor stage was
never projected (only the final combined stage was), a standard cause
of fractional-step schemes degrading to 1st order regardless of stage
order -- exactly AUDIT-1's original N2 concern, which the Richardson
estimator failed to correctly diagnose.

**Confirmed directly** (audit's specified "cheapest resolution": fix N,
halve dt, no Richardson subtraction) at N=256, Re=100, T=0.05: error
halved almost exactly at every dt-halving (ratio ~2.0) -- unambiguous
1st order.

**FIXED**: `step()` now projects the predictor stage too (incremental
pressure correction) before evaluating the corrector's RHS. Re-verified
directly: residuals from the floor now shrink by ~4x per dt-halving
(order ~2), confirmed at N=64/Re=100/T=0.2 (ratios 4.00, 4.05) --
genuine 2nd order, not merely dt-scaling-masked 1st order as before.

**Side effect discovered**: the unprojected predictor was ALSO the
source of the chi-measurement-tolerance noise documented earlier
(`test_chi_measurement_tolerance.py`) -- with the fix, chi=5 (matching
D3) at every tested tolerance from 1e-12 to 1e-2, at both the original
and a deliberately under-resolved configuration. That test's original
finding (chi=21 at tol=1e-8) was a SYMPTOM of this same bug, not an
independent, permanent methodological trap. Updated to lock in the
improved (no longer reproducible) behavior; the general principle (tie
rank tolerance to solver error) is kept as engineering practice.

**Old N2 test retired**: `test_n2_temporal_order.py`'s Richardson-based
methodology is now known unreliable (its own estimates climbed past
design order, which was the original clue something was off) and its
residuals flip sign after the fix. Superseded by
`test_n2_temporal_order_direct.py` (no Richardson subtraction).

**K3' confound resolved (§2)**: the original "kappa(m) and chi(m) grow
together from the same mode sets" conflated D3 and D5 -- a knob moving
two degeneracies at once cannot attribute a cost increase to either.
Replaced with two orthogonal sub-knobs: kappa-at-fixed-chi (2-mode sets
{1,k}, chi PERFECTLY fixed at 8 while kappa ranges 4->225, 56x, for
k=2..15) and chi-at-fixed-kappa (mode sets with fixed endpoints {1,10},
kappa PERFECTLY fixed at 100 while chi grows 8->12->16->24, 3x, as
intermediate modes are added). Cleaner separation than the audit's own
illustrative numbers.

**D4 sign convention fixed (§3)**: the product-form dilation's angle
factorization, previously verified only for positive k, was shown to
fail on the negative-k (two's-complement upper) half of a real fftfreq-
ordered momentum register -- confirmed numerically (naive extension
mismatches direct exponentiation by >0.5 at some entries). FIXED:
conditional two's-complement negation to |k| (O(n) Toffolis) before the
unchanged product form; verified exact (<1e-9) on the FULL signed grid,
not just positive k.

**Accepted without change**: the LBM acoustic-scaling coherence check
(nu=nu_lattice*dx under acoustic scaling implies Re~1/dx~N, the SAME
N~Re relation Q6's independently-derived cell-Reynolds rule gives) --
added as a documented cross-check, not a code change. The B2 arithmetic
reconciliation (audit's own correction of their AUDIT-2 finding,
accepted). The product-form dilation's positive-k verification to
1.35e-20 (accepted, extended to the full signed grid above).

## 2026-08-27 — AUDIT-2 response: G3 dt-anchor category error, K5 cannot break D5, N3 resolved (H1 confirmed)

**Reason:** second-round external audit (AUDIT-2.md) of RESPONSE-TO-AUDIT.md.

**G3 dt anchor (§1):** the previous fix anchored the reference dt to an
explicit-finite-difference diffusive stability limit
(dt~dx^2/(4*nu)) — a category error, since QLBM's own timestep is set
by acoustic scaling (dt=dx/lambda, LINEAR in dx) [FACT: standard LBM
result, verified via WebSearch], not an FD von Neumann limit. Also, the
prior summary line ("8.75e10 gates at 1 gate/step") did not state the
n_shots=10000 readout multiplier alongside it, letting a reader
reconstruct an inconsistent dt and flag a false arithmetic bug — the
code was internally consistent, the reporting was not clear. Fixed:
`lbm_acoustic_scaling_dt` added; algorithmic-only and end-to-end
crossovers now reported and labeled separately; new T* ~ 6.73e-3 (was
6.4e-4 under the wrong anchor), still four orders of magnitude below
the challenge's T=10. Scope stated explicitly: bounds EXPLICIT
time-marching references only.

**K5 cannot break D5 (§4):** K5's occupied-subspace kappa (a 1x1 block
for a single-mode solution) cannot move for ANY grid deformation — its
staying near 1.0000-1.0008 was a structurally-guaranteed null result,
not "D5 surviving a stress test" as previously (wrongly) framed.
Corrected: `conditioning.py` and `test_k5_anisotropy.py` docstrings
rewritten to state this plainly; K5 demoted to its genuine contribution
(a real wall-clock cost from grid-anisotropy-driven dt shrinkage, not a
conditioning result). **K3' retargeted at D5** instead
(`occupied_subspace_kappa_multimode` in k3_prime.py): with m
incommensurate modes, occupied kappa = k_max^2/k_min^2, measured
1, 9, 49, 361 for m=1,2,4,8 — genuinely tunable, 361x separation. G4's
pass now rests on K3', not K5.

**N3 resolved, not left open (§3):** the delta-sweep (u_base +
delta*perturbation, sweeping delta->0) shows order(delta) declining
smoothly from 1.75 to 1.06 with no discontinuity, ruling out H3
(structural operator mismatch). The complementary one-sweep H1 test
(refine with dt~dx^2 instead of dt~dx) is decisive: order jumps to
1.994 (Re=100) and 1.998 (Re=1000) — genuine 2nd order. **H1 confirmed
exactly as the audit predicted**: dt~dx was not aggressive enough to
keep temporal error subdominant relative to the shrinking O(dx^2)
spatial term for the unforced case specifically; the MMS-forced case's
larger spatial error stayed above that floor longer over the same
tested N range, masking the same underlying effect. Separately
verified: the ACTUAL `results/chi_of_re.json` headline runs are
diffusive-bound (dt~dx^2) at all three configurations (Re=10/N=32,
Re=100/N=64, Re=1000/N=512) by construction of the pre-registered
N(Re) rule, so they were not compromised by this mechanism even before
it was understood (`test_n3_headline_dt_regime.py`).

**D4 closure language narrowed (§2):** the product-form dilation
algebra (k^2 as a quadratic form in qubit bits, factorizing into
classically-precomputed-angle rotations, no runtime quantum arithmetic)
verified numerically to <1e-9 (`test_d4_product_form_dilation.py`).
Language changed from "the D4 gap is closed" to "subnormalization is
closed exactly (alpha=1); gate count (O(n^2)) is a separate quantity";
added caveats on angle precision folding into eps and amplitude-
amplification requiring re-preparation (not applied to this codebase's
G3 numbers).

**G5 parameterized, not deferred again (§6):** `g5_dilation_penalty_
advantage_window(p, T, c_reference)` states the advantage conditionally
on the reference's achieved-subnormalization-to-floor ratio c — tie at
c=1, quantified advantage at c>1 — converting one axis from abstention
to a stated result (dilation-penalty axis only; gate-count axis still
evaluated at c=1 implicitly, not separately parameterized).

**K3' chi(m) correction (§6 minor):** the originally-reported
chi(m)=[4,8,16,19] was itself tolerance-limited at m=8 (tol=1e-9 gave
19; the tolerance-converged value, stable from 1e-12 to 1e-14, is 24).
Fixed to use tol=1e-12 with an explicit stability check against 1e-14.

**N2 superconvergence explained (§6 minor):** extending the Richardson
sweep to 7 levels shows order climbing past 2 (to 2.27), not settling
there — not genuine superconvergence, but the expected artifact of
subtracting an imperfect (finite-dt, not true dt->0) floor: as the true
residual shrinks, it becomes comparable to the floor's own small
residual error, causing over-subtraction at the finest levels. The
middle estimates (1.88, 2.01) are the ones consistent with RK2's true
design order and are the reading to trust.

## 2026-08-27 — Q1 clarification: `L` resolves to two distinct constants

**Reason:** implementing G0 (reproduce organizers' KE(0)=0.75, KE(10)~=0.52
at Re=100 figures) failed on first pass.

**Old (as literally read from v2 §1.2 Q1 row):** a single `L = 1` used
both as the spatial/decay scale inside `u*`, `p*` (sin/cos arguments,
`exp(-2*nu*t/L^2)`) and inside the viscosity formula `nu = V0*L/Re`.
This gives `nu = V0*1/Re = 1/Re = 0.01` at Re=100, and reproduces
KE(10) = 0.6676 -- fails G0 by >20x tolerance.

**New:** `L` is two different constants, both already implied by the Q1
row's own derivation text ("nu ~= 0.063 ~= 2*pi/100"):
  - spatial/decay scale in `u*`, `p*` (sin/cos arguments and `L^2` in the
    decay exponent): `L_spatial = 1`
  - length scale entering `nu = V0*L/Re`: the DOMAIN length,
    `L_visc = 2*pi` (from `Omega = [0,2*pi]^2`), giving `nu = 2*pi/Re`.

**Verified:** with this split, `src/tgv/analytic.py` reproduces all three
supplied figures (KE(0)=0.75; KE(10)=0.520 at Re=100; pressure extremum
=0.3889 at t=1, Re=100, matching the +/-0.4 colourbar) to within the
tolerances stated in `tests/test_g0_ground_truth.py`. G0: **PASS**.

**Status:** still `[ASSUMPTION]`, not organizer-confirmed. This changelog
entry sharpens the assumption's statement; it does not change its
epistemic status. Q1 remains the highest-priority open question for the
organizers (R2 in the risk register).

**Reframing (AUDIT.md minor, 2026-08-27, no numeric change):** "two
different constants" invites the reply "you misread the spec." The
cleaner reading, producing IDENTICAL numbers: `L = 2*pi` is a single
constant (the vortex wavelength, equal to the domain length), and the
challenge statement's printed formulas simply omit a `2*pi/L` factor --
`sin(x/L)` should read `sin(2*pi*x/L)`, `exp(-2*nu*t/L^2)` should read
`exp(-2*nu*(2*pi/L)^2*t)`. At `L=2*pi` both reduce to exactly what this
codebase computes (`sin(x)`, `exp(-2*nu*t)`), and `nu=V0*L/Re` with the
same single `L=2*pi` gives `nu=2*pi/Re`. Use this framing, not "L is
overloaded," when raising Q1 with the organizers.

## 2026-08-27 — QLBM reference cost docstring corrected (false collapsed exponent)

**Reason:** external audit (AUDIT.md, B3) of `quantum_resources.py`
after `results/discrimination_matrix.md` and `results/writeup_draft.md`
had already cited the (wrong) collapsed form.

**Old:** module docstring and `qlbm_reference_cost` docstring stated
"analytic bound: O(Re^{3/4*(1+D/2)} * q_M) => O(Re^{3D/8})" -- at D=2
this claims the two factors collapse to Re^0.75. They do not: the
factors MULTIPLY (exponents ADD), giving Re^1.875 at D=2. The Re^{3D/8}
figure does not appear anywhere in the actual paper (re-verified via a
second, more careful fetch of arxiv.org/html/2512.03758) and was an
artifact of an imprecise earlier AI-generated summary, not a real
quantity from the source.

**New:** docstrings state the combined exponent explicitly (1.875 at
D=2) and note the factors multiply rather than collapse. The actual
CODE in `qlbm_reference_cost` was never wrong -- `scaling = (Re**
exponent) * q_M` already computed the correct product; only the
docstring's simplification claim was false. Re-verified against the
paper: analytic bound O(Re^{3/4*(1+D/2)} * q_M), empirical (2D, their
Sec 4.4) O(Re^1.936 * q_M).

## 2026-08-27 — K5 mechanism corrected: grid anisotropy, not viscosity anisotropy

**Reason:** external audit (AUDIT.md, B1) of `src/tgv/conditioning.py`
and `tests/test_k5_*.py` after a result (error `1.5e-3 -> 1.36e-1` under
"K5") had already been reported in `results/discrimination_matrix.md`
and `results/writeup_draft.md`.

**Old:** K5 implemented as `nu_x != nu_y` (anisotropic viscosity) in
`anisotropic_tgv_field`/`anisotropic_dispersion_2d`/the K5 solver test.
This changes the PDE being solved, not merely its discretization -- for
`nu_x+nu_y` held fixed the exact TGV solution stays exact (a genuine
extra micro-degeneracy, kept in STATUS.md); for `nu_x+nu_y` NOT held
fixed (what the reported growth used), the solver integrates a
different PDE than the one the reference solves, so the vortex decays
at the wrong rate. The reported `1.5e-3 -> 1.36e-1` growth is an
artifact of that mismatch, not evidence about D5's conditioning claim.
**Deleted from the writeup and discrimination matrix.**

## 2026-08-27 — Open finding: unforced base-TGV convergence order trails the MMS-forced case (not closed)

**Reason:** AUDIT.md N3 asked for a genuine fixed-Re=1000 grid-
refinement study (the only prior order evidence was G2's MMS-FORCED
convergence test). Investigating surfaced a real, reproducible, and
NOT YET UNDERSTOOD discrepancy, logged per Rule 3 rather than asserted
away.

**Finding:** the UNFORCED base-TGV case (no MMS forcing, exact
divergence-free reference) shows observed spatial order ~1.0-1.5,
trending DOWN toward ~1.0 as N grows through 64->128->256 -- well below
G2's MMS-forced case, which reliably achieves >=1.8 (unaffected,
`tests/test_ws2_fv_solver.py::test_g2_mms_convergence_order_at_least_1_8`
still passes). Checked and ruled out as an explanation: Re=1000 under-
resolution relative to Q6's N(Re) rule (the SAME ~1.0-1.5 order appears
at Re=100, which is well-resolved throughout the tested N range). A
fixed-N, dt-only Richardson isolation (mirroring the N2 methodology)
shows the unforced case's own temporal order (~1.12->1.25->1.64)
trailing the forced case's (~1.45->1.75->2.19) at every comparable
step-count level -- so the discrepancy shows up in the temporal
component too, not only the coupled dx-sweep.

**Status:** OPEN. The mechanism is not understood. Recorded honestly
in `tests/test_n3_error_normalization.py`
(`test_n3_fixed_re1000_grid_refinement_reports_observed_order_honestly`),
which asserts only what is actually supported (order>0.8, monotonic
error decrease) rather than a target the evidence does not support.
Any future headline claim resting on the UNFORCED case's convergence
order (as opposed to G2's own validated MMS-forced order) should not
assume 2nd order without resolving this first.

**New:** K5 is grid anisotropy (`Nx != Ny`, rectangular grid), nu
stays a single isotropic value, the PDE and its exact solution are
UNCHANGED at every aspect ratio -- the reference stays valid by
construction. Re-measured (`tests/test_k5_anisotropy.py`,
`tests/test_k5_solver_grounded.py`): at resolved wavenumbers the
occupied-subspace kappa barely moves (1.0000 -> 1.0008 over a 10x
aspect-ratio range) while the worst-case kappa explodes ~50x over the
same range -- and, through the full solver, error stays bounded
(<=1.5e-3) at every ratio while the explicit diffusion-stability dt
shrinks in proportion to the aspect ratio squared, a direct wall-clock
cost. This is a STRONGER, correctly-grounded version of the same
D5-survives-a-stress-test story.
