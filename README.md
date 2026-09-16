# TGX — Taylor-Green Vortex, Quantum + Tensor-Network

Submission codebase for the 2026 Global Quantum + AI Challenge (Airbus track): the 2D convecting Taylor-Green Vortex, a quantum-algorithm approach with a working compressed tensor-network solver as a secondary deliverable.

Built by **AiQC — AI & Quantum Community**.

## What's here

- `src/tgv/` — the codebase: analytic solution, classical finite-volume solver, fast-forwarding / exact-diagonalization quantum construction, non-unitarity dilation (D4), quantum resource analysis, and the compressed quantics/tensor-train solver (`qtt_solver.py`).
- `tests/` — 96 automated tests covering every claim made in the report.
- `six-degeneracies-report.html` — the full technical report: six degeneracies (D1–D6) proven independently, a six-knob instrument (K1–K6) that stresses each one, gate-by-gate verification, and a bug/audit log summarizing three rounds of external review.
- `2026-Global-Quantum-AI-Challenge-Airbus-Submission.tex` / `.pdf` — the submitted paper, mapped to the organizers' scored criteria and citing this repository directly.
- `protocol.md` / `protocol.sha256` — the frozen pre-registration and its checksum, locked before any result existed.
- `CHANGELOG-protocol.md` — every change made to that protocol after the fact, logged with date, reason, old and new value.
- `scripts/` — reproduction scripts (organizer figure reproduction, Q1 triangulation, G5 phase diagram, figure generation for the PDF).
- `results/` — measured data backing the report's tables and figures.

## Running it

```bash
pip install numpy scipy sympy matplotlib
python -m pytest tests/ -q
```

## Status

96/96 tests passing. Three audit rounds closed. Blocked on organizer confirmation of a handful of open items listed in the report's Appendix D and the PDF's "Open items" section (domain/length-scale convention, dual-track submission, Re grid scope, deadline/format).
