# Pre-registered Evaluation Protocol

Frozen per constitution v2 Rule 3 / §9, before any solver run. Any change
after this point is logged in `CHANGELOG-protocol.md` with date, reason,
old value, new value — never silently retuned.

## Parameters (Q1, as clarified in CHANGELOG-protocol.md 2026-08-27)
- Domain: Omega = [0, 2*pi]^2, doubly periodic (Q7)
- Spatial/decay length scale: L = 1
- Viscosity: nu = V0 * domain_length / Re = 2*pi / Re
- V0 = 1, Uc = 1, Vc = 0, rho = 1, p0 = 0

## Grids (§9.1)
- Re in {10, 100, 1e3, 1e4, 1e5, 1e6}. All six reported, or absence
  explained.
- N(Re) = 2^ceil(log2(max(32, Re/2)))  (Q6, cell-Reynolds rule Re_D <= 2)
- One fixed-N sweep reported separately as a stability diagnostic (F7).
- Final time T = 10 (matches organizers' KE figure); error also reported
  at t = 1 (matches organizers' field figures).
- Truncation tolerances: {1e-6, 1e-8, 1e-10, 1e-12}; rank cap >= 32 always
  (leakage guard — a cap of exactly 5 fits the answer instead of
  discovering it).
- Knobs K1-K6 swept at lambda in {0, 0.25, 0.5, 1.0}; lambda=0 must
  reproduce the challenge case exactly.

## Error definition (Q3)
Relative L2 on the velocity vector, against CELL AVERAGES (not point
samples) of the exact solution, at final T and time-max over [0,T]:
  eps2(t) = ||u_h - u*_avg||_2 / ||u*_avg||_2

## Baselines (§9.2) — all three, every run
1. Dense 2nd-order FV, explicit, uncompressed (isolates TT machinery)
2. Pseudo-spectral, dealiased (honest strong classical bar)
3. 2-mode Galerkin ("cheat solver": exact, O(1), any Re)

## Verification (§9.3)
MMS grid-convergence study; report observed vs design order of accuracy
for every scheme variant.

## Invariants checked every run (§9.4)
- max|div(u_h)| within tolerance
- energy drift bounded and reported
- Galilean check: (Uc,Vc)=(0,0) vs (1,0) give identical error up to
  translation (Q1 consequence of D1)
- TGV mode-structure symmetry preserved

## Timing (§9.5)
Wall-clock: single-threaded, warm-up excluded, median of 5 runs, machine
spec recorded (see `src/tgv/timing.py`). Quantum costs: gate/T-counts
only, never wall-clock.

## Decision gates (§11)
G0 (Day 1): reproduce organizers' KE(0)=0.75, KE(10)~=0.52 @ Re=100 —
  **PASSED 2026-08-27**, see CHANGELOG-protocol.md.
G1 (end WS1): >=4 of 6 degeneracies (D1-D6) survive symbolic derivation
  and literature check, including D1 and D4.
G2 (end WS2): each surviving degeneracy confirmed numerically; MMS order
  >= 1.8.
G3 (mid WS4): quantum algorithm end-to-end query count at K1=1 must not
  exceed time-marching QLBM at equal accuracy.
G4 (end WS3): >=2 methods separated by >1 order of cost scaling on
  >=1 knob.
G5 (end WS4): report advantage window either way; never manufacture one.

---
Frozen: 2026-08-27. Hash of this file at freeze time recorded in
`protocol.sha256`.
