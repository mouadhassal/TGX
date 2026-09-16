"""
Generate the figures for PHASE1-SUBMISSION.pdf, from real project data only:
results/chi_of_re.json (dense solver measurements) and a fresh run of
qtt_solver_demo's underlying functions (compressed solver measurements).
No numbers are invented here; everything plotted was computed elsewhere in
this repo and is just being re-read/re-run and drawn.
"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tgv.config import TGVParams
from tgv.qtt_solver import build_field_qtt, qtt_memory_bytes

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "pdf.fonttype": 42,   # embed as real TrueType, not Type 3 bitmap
    "ps.fonttype": 42,
})

with open(os.path.join(os.path.dirname(__file__), "..", "results", "chi_of_re.json")) as f:
    data = json.load(f)

measured = data["measured"]
Re_m = [d["Re"] for d in measured]
wall_m = [d["wall_s"] for d in measured]
err_m = [d["rel_l2_error"] for d in measured]
dense_bytes_m = [d["dense_bytes"] for d in measured]
qtt_bytes_m = [d["quantics_bytes"] for d in measured]

# --- Figure 1: classical dense-solver wall-clock vs Re ---
fig, ax = plt.subplots(figsize=(4.6, 3.2))
ax.loglog(Re_m, wall_m, "o-", color="#0C7A88", linewidth=1.6, markersize=5)
offsets = [(0, 10), (0, 10), (-22, -4)]
for (re, w), off in zip(zip(Re_m, wall_m), offsets):
    ax.annotate(f"{w:.1f}s", (re, w), textcoords="offset points", xytext=off,
                ha="center", fontsize=8, color="#333333")
ax.set_xlabel("Re")
ax.set_ylabel("wall-clock (s), T=10, single-threaded")
ax.set_title("Classical dense-solver wall-clock vs Re")
ax.set_ylim(top=ax.get_ylim()[1] * 3)
ax.grid(True, which="both", alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_wallclock.pdf"))
plt.close(fig)

# --- Figure 2: memory / qubit count vs Re, dense vs compressed, extended
#     with real Re=10^6 numbers from the QTT solver (not extrapolated) ---
Re_ext = [1e4, 1e5, 1e6]
dense_bytes_ext = []
qtt_bytes_ext = []
for Re, n_bits in zip(Re_ext, (14, 17, 19)):
    p = TGVParams(Re=Re)
    N = 2 ** n_bits
    qtt = build_field_qtt(p, t=10.0, n_bits=n_bits)
    qtt_bytes_ext.append(qtt_memory_bytes(qtt))
    dense_bytes_ext.append(2 * N * N * 8)

Re_all = Re_m + Re_ext
dense_all = dense_bytes_m + dense_bytes_ext
qtt_all = qtt_bytes_m + qtt_bytes_ext

fig, ax = plt.subplots(figsize=(4.6, 3.2))
ax.loglog(Re_m, dense_bytes_m, "o-", color="#0C7A88", linewidth=1.6, markersize=5, label="dense storage (measured)")
ax.loglog(Re_ext, dense_bytes_ext, "o--", color="#0C7A88", linewidth=1.2, markersize=5, markerfacecolor="white",
          label="dense storage (computed size, never allocated)")
ax.loglog(Re_all, qtt_all, "s-", color="#B96A1A", linewidth=1.6, markersize=5, label="QTT compressed (measured, all points run)")
ax.set_xlabel("Re")
ax.set_ylabel("bytes")
ax.set_title("Memory / qubit count vs Re")
ax.legend(fontsize=7.5, loc="upper left")
ax.grid(True, which="both", alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_memory.pdf"))
plt.close(fig)

# --- Figure 3: relative L2 error vs Re ---
fig, ax = plt.subplots(figsize=(4.6, 3.2))
ax.loglog(Re_m, err_m, "o-", color="#AD4139", linewidth=1.6, markersize=5)
offsets = [(0, 10), (0, 14), (0, 10)]
for (re, e), off in zip(zip(Re_m, err_m), offsets):
    ax.annotate(f"{e:.2e}", (re, e), textcoords="offset points", xytext=off,
                ha="center", fontsize=8, color="#333333")
ax.set_xlabel("Re")
ax.set_ylabel("relative L2 error at T=10")
ax.set_title("Error scaling vs Re (post temporal-order fix)", fontsize=9.5)
ax.set_ylim(top=ax.get_ylim()[1] * 2.5)
ax.grid(True, which="both", alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_error.pdf"))
plt.close(fig)

# --- Figure 4: compression ratio vs Re, all six pre-registered points ---
fig, ax = plt.subplots(figsize=(4.6, 3.2))
ratio = [d / q for d, q in zip(dense_all, qtt_all)]
ax.semilogy(range(len(Re_all)), ratio, "D-", color="#2E7D4F", linewidth=1.6, markersize=6)
ax.set_xticks(range(len(Re_all)))
ax.set_xticklabels([f"$10^{{{int(np.log10(r))}}}$" for r in Re_all])
ax.set_xlabel("Re")
ax.set_ylabel("dense / compressed size ratio")
ax.set_title("QTT compression ratio, full pre-registered Re grid", fontsize=9.5)
ax.grid(True, which="both", alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_compression.pdf"))
plt.close(fig)

print("Wrote figures to", OUT)
for fn in ("fig_wallclock.pdf", "fig_memory.pdf", "fig_error.pdf", "fig_compression.pdf"):
    print(" -", fn)

print()
print("Table data (for the LaTeX table, copy-checked against this printout):")
print(f"{'Re':>10} {'N':>10} {'dense bytes':>14} {'QTT bytes':>10} {'ratio':>12}")
for re, d, q in zip(Re_all, dense_all, qtt_all):
    print(f"{re:10.0f} {'':>10} {d:14,d} {q:10,d} {d/q:12,.1f}x")
