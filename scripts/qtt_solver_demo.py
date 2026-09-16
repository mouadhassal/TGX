"""
Real numbers for the report's new §1b (compressed solver, item #2), all
produced by src/tgv/qtt_solver.py -- not claimed, not extrapolated.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tgv.config import TGVParams
from tgv.analytic import mean_kinetic_energy
from tgv.qtt_solver import build_field_qtt, mean_kinetic_energy_qtt, qtt_memory_bytes


def main():
    print("QTT compressed solver: real, measured numbers (qtt_solver.py)\n")
    print(f"{'Re':>10} | {'n_bits':>6} | {'N':>10} | {'dense (u,v) bytes':>18} | {'QTT bytes (measured)':>20} | {'compression':>12}")
    print("-" * 90)
    for Re, n_bits in [(100, 6), (1000, 9), (10000, 14), (100000, 17), (1_000_000, 19)]:
        N = 2 ** n_bits
        p = TGVParams(Re=Re)
        qtt = build_field_qtt(p, t=10.0, n_bits=n_bits)
        qtt_bytes = qtt_memory_bytes(qtt)
        dense_bytes = 2 * N * N * 8
        print(f"{Re:10d} | {n_bits:6d} | {N:10d} | {dense_bytes:18,d} | {qtt_bytes:20,d} | {dense_bytes/qtt_bytes:12,.1f}x")

    print()
    print("KE readout accuracy, entirely from the compressed representation")
    print("(no dense array formed at any of these points, including the last):")
    print(f"{'Re':>10} | {'t':>5} | {'KE (QTT readout)':>20} | {'KE (closed form)':>20} | {'abs diff':>12}")
    print("-" * 80)
    for Re, n_bits in [(100, 6), (1_000_000, 19)]:
        p = TGVParams(Re=Re)
        for t in (0.0, 5.0, 10.0):
            ke_qtt = mean_kinetic_energy_qtt(p, t, n_bits)
            ke_ref = float(mean_kinetic_energy(p, t))
            print(f"{Re:10d} | {t:5.1f} | {ke_qtt:20.12f} | {ke_ref:20.12f} | {abs(ke_qtt-ke_ref):12.3e}")


if __name__ == "__main__":
    main()
