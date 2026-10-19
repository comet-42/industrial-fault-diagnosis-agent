from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional

from .models import DiagnosisReport, Evidence, SignalRecord
from .tools import (extract_features, inspect_signal, retrieve_knowledge,
                    run_diagnostic_model, validate_diagnosis)


class FaultDiagnosisAgent:
    """Evidence-grounded workflow agent.

    The planner is intentionally deterministic for reproducible evaluation. An
    LLM planner can be added later, but every action still passes through the
    typed tools and validation gate.
    """

    def diagnose(self, signal: SignalRecord, *, run_id: Optional[str] = None) -> DiagnosisReport:
        inspection, trace_1 = inspect_signal(signal)
        features, trace_2 = extract_features(signal)
        prediction, trace_3 = run_diagnostic_model(features)
        knowledge, trace_4 = retrieve_knowledge(prediction["label"])
        validation, trace_5 = validate_diagnosis(prediction["label"], prediction["confidence"], inspection, features)
        label = prediction["label"] if validation["valid"] else "uncertain"
        confidence = prediction["confidence"] if validation["valid"] else min(prediction["confidence"], 0.45)
        severity = {"normal": 0.05, "phase_open": 0.90, "interturn_short": 0.85,
                    "demagnetization": 0.65, "sensor_bias": 0.40}.get(label, 0.50)
        uncertainty = list(validation["reasons"]) if not validation["valid"] else []
        return DiagnosisReport(
            diagnosis=knowledge["claim"] if validation["valid"] else "Diagnosis withheld pending data quality or engineering review.",
            fault_family=label, severity=severity, confidence=confidence,
            recommended_action=knowledge["recommended_action"] if validation["valid"] else "Collect a valid window and escalate to an engineer.",
            evidence=[Evidence("signal_inspection", "input window is finite", inspection["finite"], 1.0),
                      Evidence("feature_extractor", "channel RMS", {k: round(features[k]["rms"], 4) for k in ("i_a", "i_b", "i_c")}, 0.95),
                      Evidence("diagnostic_model", "predicted class", prediction["label"], prediction["confidence"]),
                      Evidence(knowledge["source"], "engineering interpretation", knowledge["claim"], 0.90)],
            tool_trace=[trace_1, trace_2, trace_3, trace_4, trace_5],
            uncertainty=uncertainty, run_id=run_id or self._run_id(signal),
        )

    @staticmethod
    def _run_id(signal: SignalRecord) -> str:
        payload = json.dumps(signal.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(payload.encode()).hexdigest()[:12]
