"""
Wall-clock timing harness (constitution v2 Rule 6, §9.5): single-threaded
unless stated, warm-up excluded, median of 5 runs, machine spec recorded
in every timing artifact. Quantum costs are NEVER timed here -- they are
gate/T-counts (WS4), reported separately.
"""
import json
import platform as _platform
import statistics
import time
from dataclasses import dataclass, asdict, field
from typing import Callable


@dataclass
class MachineSpec:
    platform: str = field(default_factory=_platform.platform)
    processor: str = field(default_factory=_platform.processor)
    python_version: str = field(default_factory=_platform.python_version)
    machine: str = field(default_factory=_platform.machine)


@dataclass
class TimingResult:
    label: str
    median_s: float
    all_s: list
    n_warmup: int
    n_runs: int
    machine: MachineSpec

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=2)


def time_median(fn: Callable, label: str, n_warmup: int = 1, n_runs: int = 5) -> TimingResult:
    """
    Run fn() n_warmup times (discarded), then n_runs times, and record the
    median wall-clock time. fn should take no arguments (use a closure /
    functools.partial) and should be single-threaded unless the caller
    documents otherwise in `label`.
    """
    for _ in range(n_warmup):
        fn()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return TimingResult(
        label=label,
        median_s=statistics.median(times),
        all_s=times,
        n_warmup=n_warmup,
        n_runs=n_runs,
        machine=MachineSpec(),
    )
