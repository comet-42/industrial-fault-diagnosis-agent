from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from .models import Evidence, SignalRecord, ToolTrace


def _trace(name: str, purpose: str, inp: Dict[str, Any], out: Dict[str, Any]) -> ToolTrace:
    return ToolTrace(name, purpose, inp, out)


def inspect_signal(signal: SignalRecord) -> Tuple[Dict[str, Any], ToolTrace]:
    lengths = {k: len(v) for k, v in signal.channels.items()}
    finite = all(np.isfinite(v).all() for v in (np.asarray(x) for x in signal.channels.values()))
    report = {"channels": list(signal.channels), "n_samples": signal.n_samples,
              "sample_rate_hz": signal.sample_rate_hz, "lengths": lengths,
              "finite": bool(finite), "duration_s": signal.n_samples / signal.sample_rate_hz if signal.sample_rate_hz else 0.0}
    return report, _trace("inspect_signal", "Check schema, sampling and missing values before analysis.",
                         {"n_channels": len(signal.channels)}, report)


def extract_features(signal: SignalRecord) -> Tuple[Dict[str, Any], ToolTrace]:
    feats: Dict[str, Any] = {}
    for name, values in signal.channels.items():
        x = np.asarray(values, dtype=float)
        rms = float(np.sqrt(np.mean(x * x)))
        centered = x - x.mean()
        crest = float(np.max(np.abs(centered)) / (np.sqrt(np.mean(centered * centered)) + 1e-9))
        spectrum = np.abs(np.fft.rfft(centered))
        freqs = np.fft.rfftfreq(len(x), 1.0 / signal.sample_rate_hz)
        peak = int(np.argmax(spectrum[1:]) + 1) if len(spectrum) > 1 else 0
        feats[name] = {"mean": float(x.mean()), "std": float(x.std()), "rms": rms,
                       "crest_factor": crest, "dominant_hz": float(freqs[peak]),
                       "high_frequency_ratio": float(spectrum[freqs > 200].sum() / (spectrum.sum() + 1e-9))}
    feats["imbalance"] = float(np.std([feats[k]["rms"] for k in ("i_a", "i_b", "i_c")]))
    return feats, _trace("extract_features", "Compute interpretable time/frequency descriptors.",
                         {"channels": list(signal.channels)}, {"feature_count": 6 * len(signal.channels) + 1})


def run_diagnostic_model(features: Dict[str, Any]) -> Tuple[Dict[str, Any], ToolTrace]:
    """A transparent baseline standing in for the user's frozen TCN.

    In production, replace this function with an ONNX Runtime adapter while
    retaining the same output contract.
    """
    rms = {k: features[k]["rms"] for k in ("i_a", "i_b", "i_c")}
    means = {k: abs(features[k]["mean"]) for k in rms}
    hf = {k: features[k]["high_frequency_ratio"] for k in rms}
    if rms["i_b"] < 1.0:
        label, score = "phase_open", 0.97
    elif max(hf.values()) > 0.22:
        label, score = "interturn_short", 0.83
    elif max(means.values()) > 0.75:
        label, score = "sensor_bias", 0.88
    elif max(rms.values()) < 4.8:
        label, score = "demagnetization", 0.82
    else:
        label, score = "normal", 0.79
    result = {"label": label, "confidence": score,
              "class_probabilities": {label: score, "other": 1.0 - score},
              "model": "transparent-baseline-v1"}
    return result, _trace("run_diagnostic_model", "Infer fault class from engineered descriptors.",
                         {"feature_count": len(features)}, {"label": label, "confidence": score})


def retrieve_knowledge(label: str) -> Tuple[Dict[str, Any], ToolTrace]:
    kb = {
        "normal": ("No characteristic fault signature detected.", "Continue routine monitoring; verify under load."),
        "phase_open": ("One phase current is near zero, consistent with an open-phase condition.", "Stop or derate safely; inspect phase wiring and inverter protection."),
        "interturn_short": ("Elevated high-frequency current content suggests a turn-to-turn short.", "Schedule controlled shutdown and insulation inspection."),
        "demagnetization": ("Balanced but reduced current amplitude is consistent with flux loss.", "Check magnets, temperature history and torque margin."),
        "sensor_bias": ("A persistent channel mean offset suggests a current-sensor bias.", "Check sensor zero, ADC offset and calibration before mechanical intervention."),
    }
    claim, action = kb.get(label, ("No knowledge entry available.", "Escalate for engineering review."))
    out = {"claim": claim, "recommended_action": action, "source": "pmsm-fault-kb-v1"}
    return out, _trace("retrieve_knowledge", "Ground the model label in an actionable engineering note.",
                       {"label": label}, {"source": out["source"]})


def validate_diagnosis(label: str, confidence: float, inspection: Dict[str, Any], features: Dict[str, Any]) -> Tuple[Dict[str, Any], ToolTrace]:
    reasons: List[str] = []
    valid = bool(inspection["finite"] and inspection["n_samples"] >= 100 and confidence >= 0.60)
    if not inspection["finite"]:
        reasons.append("non-finite input")
    if inspection["n_samples"] < 100:
        reasons.append("window shorter than 100 samples")
    if confidence < 0.60:
        reasons.append("model confidence below threshold")
    if label == "phase_open" and features.get("i_b", {}).get("rms", 1.0) >= 1.0:
        valid = False
        reasons.append("phase-open rule not supported by channel RMS")
    out = {"valid": valid, "reasons": reasons or ["all quality gates passed"],
           "escalate": not valid}
    return out, _trace("validate_diagnosis", "Reject unsupported conclusions and trigger escalation.",
                       {"label": label, "confidence": confidence}, out)
