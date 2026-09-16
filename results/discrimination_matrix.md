# Discrimination matrix: methods x knobs

Built from `tests/test_k*.py` and `tests/test_d*.py` (all passing as of
this writeup). Every cell traces to a specific test. Cost scaling is
reported as measured/derived; `--` means not yet measured for that cell.

| Knob | Breaks | Our classical instrument (quantics FV) | Naive fast-forward (our alg., §3.1a) | Carleman+QLSA | QLBM (Jennings et al.) |
|---|---|---|---|---|---|
| lambda=0 (base TGV) | -- (verification corner) | chi=5 at Re in {10,100,1000}, exact `results/chi_of_re.json` | O(1) in T [DERIVED, D2] | R suppressed O(dx^p) [DERIVED, D1(2)] -- optimistic | O(Re^1.936 * q_M) [FACT: arXiv:2512.03758] |
| K1 (second mode, eps>0) | D1 | unforced-solver error jumps 35x+ at eps=0->0.1; exact-field chi jumps 5->9 (`test_k1_solver_grounded.py`) | LCU correction term added, cost ~ O(eps) [DERIVED] | R>=1 becomes possible again; convergence no longer guaranteed | not separately measured (K1 doesn't change QLBM's own scaling model) |
| K2 (Uc(t) time-dependent) | D2 | unforced-solver error jumps ~67x from amplitude=0 to 0.2, using a manufactured Uc(t)=Uc0+A*sin(wt) field with correctly-tracked phase (`test_k2_solver_grounded.py`) | naive single-shot fast-forward diverges from truth at the operator level (`test_k2_breaks_fastforward.py`); genuine time-marching (or Magnus-type expansion) required | unaffected (Carleman doesn't rely on time-independence the same way) | unaffected |
| K3 (spectral slope) | D3 | **CUT** -- kill criterion triggered at n=10: chi_nonlinear(8) > chi_field(6) (`test_k3_pretest.py`) | -- | -- | -- |
| K3' (m-mode superposition, zero forcing) | D3 **and** D5 | chi(m) = 4,8,16,24 for m=1,2,4,8 incommensurate modes, tolerance-converged (`test_k3_prime.py`); two ORTHOGONAL sub-knobs isolate each degeneracy: kappa-at-fixed-chi (chi held exactly at 8, kappa ranges 4->225, 56x, across k in {2,3,4,5,7,10,15}) and chi-at-fixed-kappa (kappa held exactly at 100, chi grows 8->12->16->24, 3x, as intermediate modes are added) | occupied-subspace kappa = k_max^2/k_min^2, directly tunable by construction from the same mode sets [DERIVED] | kappa-dependent QLSA cost scales directly with the (now nontrivial) occupied-subspace kappa [DERIVED] | -- |
| K4 (Uc=0) | D4 | KE-ratio decays 0.77->0.52->0.28 at T=2,5,10 vs Uc=1's floor ~0.87 (`test_k4_solver_grounded.py`); norm-ratio floor unbounded as T->inf (`test_d4_nonunitarity.py`) | dilation penalty unbounded as T->inf [DERIVED] -- restores the non-unitarity difficulty | unaffected in kind (Carleman's own issues are separate) | unaffected in kind |
| K5 (grid anisotropy, Nx != Ny) | **not D5** -- real wall-clock cost only (see note below) | error stays BOUNDED (<=1.5e-3, even improves) at every aspect ratio 1..8 -- confirms the reference stays valid, since nu is isotropic and the PDE is unchanged (`test_k5_solver_grounded.py`); stability-limited dt shrinks ~100x over a 10x aspect ratio -- the genuine cost signal | occupied-subspace kappa CANNOT move: for a single-mode solution it is the condition number of a 1x1 block, exactly 1 for every grid deformation by definition (`test_k5_anisotropy.py`) -- a structurally null result, not evidence about D5 | unaffected (K5's mechanism has no QLSA-relevant kappa to move) | -- |
| K6 (local observable) | D6 | global KE = 1 solver reduction regardless of N; n_probes independent local outputs cost n_probes x a single-observable budget (`test_k6_local_observable.py`) | extraction cost O(n_probes/eps^2) vs O(1/eps^2) for the global functional [DERIVED, D6] | same extraction-cost asymmetry applies to any algorithm's readout step | this is exactly bottleneck #4 (inefficient data extraction) that Jennings et al. name; K6 makes it concrete |

## Reading the matrix

- **Base corner (lambda=0) is the challenge's own case**: every column
  has an entry, none diverges, all three of D1/D2/D3's degeneracies
  hold simultaneously -- consistent with the "benchmark is degenerate"
  claim (constitution v2 section 2).
- **K4 and K2 are the cleanest discriminators among the "wired knobs":**
  both are near-free parameter changes, and both produce measurable,
  unambiguous separation (K2: a working algorithm vs. a broken
  shortcut; K4: a bounded vs. unbounded dilation cost).
- **G4's pass rests on K3', not K5.** K5 was originally credited with
  a >10x occupied-vs-worst-case kappa separation; a second-round audit
  showed this could not be real, since the occupied-subspace kappa for
  a single-mode solution is a 1x1 block's condition number -- exactly
  1 for ANY grid deformation, structurally. K5 was demoted to its one
  honest contribution (a real wall-clock cost from stability-limited
  dt shrinkage). K3' genuinely breaks D5 instead: two orthogonal
  sub-knobs give a clean 56x (kappa-at-fixed-chi) and 3x
  (chi-at-fixed-kappa) separation, each attributable to one metric at
  a time, from mode sets that also independently confirm D3 with a
  3-24x chi range.
- **K3 is a genuine negative result**, not a gap: it failed its own
  pre-registered kill criterion and was cut per Rule 4, honestly
  reported per Rule 3. K3' reinstates D3 coverage with zero forcing.
- **K1, K2, K3', K4, K5, K6 are all now confirmed both at the operator/
  analytic level AND wired end-to-end through the full nonlinear FV
  solver** (K3' at the exact-field level, which is the level its
  zero-forcing construction actually lives at). What's still `--`:
  QLBM's response to the K1/K3'/K5 knobs (would need re-deriving
  Jennings et al.'s scaling law under a perturbed/anisotropic operator,
  which requires their circuit-level construction, not just the
  scaling exponent we fetched).
- **G3** (quantum algorithm at K1=1 vs. time-marching QLBM, must not be
  worse): passes with a NUMERIC crossover (not merely asymptotic),
  correctly anchored to LBM's own acoustic-scaling timestep (an earlier
  anchor used an explicit-FD diffusive stability limit -- the wrong
  category of quantity for this reference). T* ~ 6.73e-3, roughly
  1500x below the challenge's own T=10; at T=10 the reference's cost
  (a generous lower bound: 1 gate/step) already exceeds ours by ~3
  orders of magnitude. We still do not know Jennings et al.'s exact
  per-step gate constant, so this beats a generous lower bound on their
  cost, not their verified exact circuit.
