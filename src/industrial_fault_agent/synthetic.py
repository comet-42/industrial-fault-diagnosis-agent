"""Public, deterministic synthetic data for demos and tests.

The generator intentionally produces no private research data. Its signatures are
simple approximations that let the agent demonstrate tool use and evaluation.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .models import SignalRecord

FAULTS = ("normal", "phase_open", "interturn_short", "demagnetization", "sensor_bias")


def generate_signal(fault: str = "normal", *, seed: int = 7, sample_rate_hz: float = 2500.0,
                    duration_s: float = 0.2, noise: float = 0.03) -> SignalRecord:
    if fault not in FAULTS:
        raise ValueError(f"unknown fault {fault!r}; choose from {FAULTS}")
    rng = np.random.default_rng(seed)
    n = int(sample_rate_hz * duration_s)
    t = np.arange(n) / sample_rate_hz
    base = 8.0 * np.sin(2 * math.pi * 50 * t) + 0.7 * np.sin(2 * math.pi * 150 * t)
    phases: Dict[str, list[float]] = {}
    for i, phase in enumerate((0.0, -2 * math.pi / 3, 2 * math.pi / 3)):
        x = 8.0 * np.sin(2 * math.pi * 50 * t + phase) + 0.7 * np.sin(2 * math.pi * 150 * t + 3 * phase)
        x += rng.normal(0, noise, n)
        phases[f"i_{'abc'[i]}"] = x.tolist()
    if fault == "phase_open":
        phases["i_b"] = (0.04 * rng.normal(size=n)).tolist()
    elif fault == "interturn_short":
        phases["i_a"] = (np.asarray(phases["i_a"]) + 2.2 * np.sin(2 * math.pi * 250 * t)).tolist()
    elif fault == "demagnetization":
        phases = {k: (0.58 * np.asarray(v)).tolist() for k, v in phases.items()}
    elif fault == "sensor_bias":
        phases["i_c"] = (np.asarray(phases["i_c"]) + 2.0).tolist()
    return SignalRecord(sample_rate_hz=sample_rate_hz, channels=phases,
                        metadata={"fault_label": fault, "source": "synthetic-demo", "seed": seed})
