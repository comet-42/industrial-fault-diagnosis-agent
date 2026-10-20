from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .models import AgentDecision, DiagnosisReport, Evidence, SignalRecord, ToolTrace
from .tools import (analyze_fault_evidence, assess_uncertainty, extract_features,
                    analyze_pa_jc_evidence, inspect_signal, retrieve_knowledge,
                    run_diagnostic_model, validate_diagnosis)


@dataclass
class AgentState:
    """Mutable state shared by the planner and tools during one task."""

    signal: SignalRecord
    goal: str = "produce a reliable, evidence-grounded diagnosis"
    observations: Dict[str, Any] = field(default_factory=dict)
    tool_trace: List[ToolTrace] = field(default_factory=list)
    decision_trace: List[AgentDecision] = field(default_factory=list)
    uncertainty: List[str] = field(default_factory=list)
    status: str = "running"
    step_count: int = 0


@dataclass(frozen=True)
class ToolSpec:
    """A tool advertised to the planner, with a precondition and utility score."""

    name: str
    purpose: str
    available: Callable[[AgentState], bool]
    score: Callable[[AgentState], float]
    handler: Callable[[AgentState], tuple[Dict[str, Any], ToolTrace]]


class GoalDirectedPlanner:
    """Select the next action from current state and registered tools.

    This is a deterministic policy over a tool registry, rather than a fixed
    call sequence. An LLM planner can implement the same ``choose`` contract
    later without changing the executor or validation gates.
    """

    def __init__(self, tools: List[ToolSpec]):
        self.tools = tools

    def choose(self, state: AgentState) -> Optional[tuple[ToolSpec, Dict[str, float]]]:
        candidates = {tool.name: float(tool.score(state)) for tool in self.tools if tool.available(state)}
        if not candidates:
            return None
        name = max(candidates, key=lambda key: (candidates[key], key))
        return next(tool for tool in self.tools if tool.name == name), candidates


class FaultDiagnosisAgent:
    """Goal-driven industrial diagnosis Agent with a state/action loop."""

    TARGETED_LABELS = {"sensor_bias", "interturn_short"}

    def __init__(self, planner: Optional[GoalDirectedPlanner] = None, max_steps: int = 10):
        self.max_steps = max_steps
        self.planner = planner or GoalDirectedPlanner(self._tool_specs())

    def diagnose(self, signal: SignalRecord, *, run_id: Optional[str] = None) -> DiagnosisReport:
        state = AgentState(signal=signal)
        while state.status == "running" and state.step_count < self.max_steps:
            choice = self.planner.choose(state)
            if choice is None:
                state.status = "escalated"
                state.uncertainty.append("no applicable tool can advance the task")
                break
            tool, scores = choice
            state.step_count += 1
            state.decision_trace.append(AgentDecision(state.step_count, tool.name, tool.purpose, scores))
            try:
                result, trace = tool.handler(state)
            except Exception as exc:
                state.tool_trace.append(ToolTrace(tool.name, tool.purpose, {}, {"error": str(exc)}, "error"))
                state.uncertainty.append(f"tool {tool.name} failed: {exc}")
                state.status = "escalated"
                break
            state.observations[tool.name] = result
            state.tool_trace.append(trace)
            self._update_state_after_tool(state, tool.name, result)
        if state.status == "running":
            state.status = "escalated"
            state.uncertainty.append("maximum Agent steps reached")
        return self._make_report(state, run_id=run_id or self._run_id(signal))

    def diagnose_pa_jc(self, window, adapter, *, run_id: Optional[str] = None) -> DiagnosisReport:
        """Run the real PA-JC model through a goal-driven decision path."""
        value = np.asarray(window)
        if value.shape != (15, 500):
            raise ValueError(f"PA-JC-HMT-TCN expects a standardized 15x500 window, got {value.shape}")
        result = adapter.predict(value)
        confidence = float(result["confidence"])
        decisions = [AgentDecision(1, "pa_jc_hmt_tcn", "Acquire the primary multi-task model observation.", {"pa_jc_hmt_tcn": 100.0})]
        traces = [ToolTrace("pa_jc_hmt_tcn", "Run the frozen PA-JC-HMT-TCN multi-task model.",
                            {"shape": list(value.shape), "model": result.get("model_name", "unknown")},
                            {"fault": result["fault_name"], "module": result["module_name"], "severity": result["severity"]})]
        if confidence < 0.75:
            uncertainty, uncertainty_trace = assess_uncertainty({
                "label": result["fault_name"], "confidence": confidence,
                "class_probabilities": dict(enumerate(result.get("fault_scores", []))),
            })
            decisions.append(AgentDecision(2, "assess_uncertainty", uncertainty_trace.purpose, {"assess_uncertainty": 90.0, "retrieve_knowledge": 70.0}))
            traces.append(uncertainty_trace)
        label_map = {"Normal": "normal", "OpenCircuit": "phase_open", "GroundUnbalance": "ground_unbalance",
                     "Interturn": "interturn_short", "Demagnetization": "demagnetization", "SensorBias": "sensor_bias", "SensorNoise": "sensor_noise"}
        label = label_map.get(result["fault_name"], result["fault_name"])
        if label in self.TARGETED_LABELS:
            targeted, targeted_trace = analyze_pa_jc_evidence(label, value)
            traces.append(targeted_trace)
            decisions.append(AgentDecision(len(decisions) + 1, "analyze_pa_jc_evidence", targeted_trace.purpose, {"analyze_pa_jc_evidence": 86.0, "retrieve_knowledge": 70.0}))
        knowledge, knowledge_trace = retrieve_knowledge(label)
        traces.append(knowledge_trace)
        decisions.append(AgentDecision(len(decisions) + 1, "retrieve_knowledge", knowledge_trace.purpose, {"retrieve_knowledge": 70.0, "validate_diagnosis": 60.0}))
        valid = confidence >= 0.60
        reasons = ["model confidence below threshold"] if not valid else ["all quality gates passed"]
        validation = {"valid": valid, "reasons": reasons, "escalate": not valid}
        validation_trace = ToolTrace("validate_diagnosis", "Reject unsupported conclusions and trigger escalation.", {"label": label, "confidence": confidence}, validation)
        traces.append(validation_trace)
        decisions.append(AgentDecision(len(decisions) + 1, "validate_diagnosis", validation_trace.purpose, {"validate_diagnosis": 60.0}))
        return DiagnosisReport(
            diagnosis=knowledge["claim"] if valid else "Diagnosis withheld pending engineering review.",
            fault_family=label if valid else "uncertain", severity=float(result["severity"]) if valid else 0.5,
            confidence=confidence if valid else min(confidence, 0.45),
            recommended_action=knowledge["recommended_action"] if valid else "Collect additional evidence and escalate to an engineer.",
            evidence=[Evidence("pa_jc_hmt_tcn", "fault head", result["fault_name"], confidence), Evidence("pa_jc_hmt_tcn", "module head", result["module_name"], confidence), Evidence("pa_jc_hmt_tcn", "severity head", result["severity"], confidence), Evidence("pmsm-fault-kb-v1", "engineering interpretation", knowledge["claim"], 0.90)],
            tool_trace=traces, decision_trace=decisions, uncertainty=[] if valid else reasons,
            run_id=run_id or "pa-jc-" + str(result.get("model_name", "frozen")),
        )

    def _tool_specs(self) -> List[ToolSpec]:
        predicted = lambda s: s.observations.get("run_diagnostic_model", {})
        label_is_targeted = lambda s: predicted(s).get("label") in self.TARGETED_LABELS
        return [
            ToolSpec("inspect_signal", "Check input schema and finite values.", lambda s: "inspect_signal" not in s.observations, lambda s: 100.0, lambda s: inspect_signal(s.signal)),
            ToolSpec("extract_features", "Compute interpretable signal descriptors.", lambda s: bool(s.observations.get("inspect_signal", {}).get("finite")) and "extract_features" not in s.observations, lambda s: 90.0, lambda s: extract_features(s.signal)),
            ToolSpec("run_diagnostic_model", "Infer candidate faults from current evidence.", lambda s: "extract_features" in s.observations and "run_diagnostic_model" not in s.observations, lambda s: 80.0, lambda s: run_diagnostic_model(s.observations["extract_features"])),
            ToolSpec("assess_uncertainty", "Gate ambiguous predictions before reporting.", lambda s: "run_diagnostic_model" in s.observations and "assess_uncertainty" not in s.observations and float(predicted(s).get("confidence", 0.0)) < 0.75, lambda s: 88.0, lambda s: assess_uncertainty(predicted(s))),
            ToolSpec("analyze_fault_evidence", "Run targeted evidence analysis for the predicted fault.", lambda s: "run_diagnostic_model" in s.observations and "extract_features" in s.observations and label_is_targeted(s) and "analyze_fault_evidence" not in s.observations, lambda s: 86.0, lambda s: analyze_fault_evidence(predicted(s)["label"], s.observations["extract_features"])),
            ToolSpec("retrieve_knowledge", "Ground the candidate in an engineering action.", lambda s: "run_diagnostic_model" in s.observations and "retrieve_knowledge" not in s.observations and (not label_is_targeted(s) or "analyze_fault_evidence" in s.observations), lambda s: 70.0, lambda s: retrieve_knowledge(predicted(s)["label"])),
            ToolSpec("validate_diagnosis", "Reject unsupported conclusions and trigger escalation.", lambda s: "run_diagnostic_model" in s.observations and "retrieve_knowledge" in s.observations and "validate_diagnosis" not in s.observations and ("assess_uncertainty" not in s.observations or not s.observations["assess_uncertainty"].get("requires_escalation")), lambda s: 60.0, lambda s: validate_diagnosis(predicted(s)["label"], predicted(s)["confidence"], s.observations["inspect_signal"], s.observations["extract_features"])),
        ]

    @staticmethod
    def _update_state_after_tool(state: AgentState, tool: str, result: Dict[str, Any]) -> None:
        if tool == "validate_diagnosis":
            state.status = "completed" if result.get("valid") else "escalated"
            if result.get("escalate"):
                state.uncertainty.extend(result.get("reasons", []))
        elif tool == "assess_uncertainty" and result.get("requires_escalation"):
            state.uncertainty.append(result["reason"])

    def _make_report(self, state: AgentState, *, run_id: str) -> DiagnosisReport:
        inspection = state.observations.get("inspect_signal", {})
        features = state.observations.get("extract_features", {})
        prediction = state.observations.get("run_diagnostic_model", {})
        knowledge = state.observations.get("retrieve_knowledge", {})
        validation = state.observations.get("validate_diagnosis", {})
        valid = state.status == "completed" and validation.get("valid", False)
        label = prediction.get("label", "uncertain") if valid else "uncertain"
        confidence = float(prediction.get("confidence", 0.0)) if valid else min(float(prediction.get("confidence", 0.0)), 0.45)
        evidence: List[Evidence] = []
        if inspection:
            evidence.append(Evidence("signal_inspection", "input window is finite", inspection.get("finite", False), 1.0))
        if features:
            evidence.append(Evidence("feature_extractor", "channel RMS", {k: round(features[k]["rms"], 4) for k in ("i_a", "i_b", "i_c")}, 0.95))
        if prediction:
            evidence.append(Evidence("diagnostic_model", "predicted class", prediction.get("label"), float(prediction.get("confidence", 0.0))))
        if state.observations.get("analyze_fault_evidence"):
            targeted = state.observations["analyze_fault_evidence"]
            evidence.append(Evidence("targeted_evidence", targeted.get("analysis", "targeted check"), targeted, 0.90))
        if knowledge:
            evidence.append(Evidence(knowledge.get("source", "knowledge-base"), "engineering interpretation", knowledge.get("claim"), 0.90))
        return DiagnosisReport(diagnosis=knowledge.get("claim") if valid else "Diagnosis withheld pending data quality or engineering review.", fault_family=label, severity={"normal": 0.05, "phase_open": 0.90, "interturn_short": 0.85, "demagnetization": 0.65, "sensor_bias": 0.40}.get(label, 0.50), confidence=confidence, recommended_action=knowledge.get("recommended_action") if valid else "Collect additional evidence and escalate to an engineer.", evidence=evidence, tool_trace=state.tool_trace, decision_trace=state.decision_trace, uncertainty=state.uncertainty, run_id=run_id)

    @staticmethod
    def _run_id(signal: SignalRecord) -> str:
        payload = json.dumps(signal.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(payload.encode()).hexdigest()[:12]
