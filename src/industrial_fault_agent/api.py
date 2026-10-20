"""Optional HTTP surface for a small local or hosted demonstration."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .agent import FaultDiagnosisAgent
from .models import SignalRecord
from .natural_language import NaturalLanguageAgent, RuleBasedTaskParser, TaskParser


def create_app(agent: FaultDiagnosisAgent | None = None, parser: TaskParser | None = None):
    """Create a FastAPI app without importing FastAPI in the core runtime."""
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the API extra with: pip install -e '.[api]'") from exc

    diagnosis_agent = agent or FaultDiagnosisAgent()
    language_agent = NaturalLanguageAgent(agent=diagnosis_agent, parser=parser or RuleBasedTaskParser())
    metrics = Counter()
    app = FastAPI(title="Industrial Fault Diagnosis Agent", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "agent": "industrial-fault-agent"}

    @app.get("/metrics")
    def get_metrics() -> dict[str, int]:
        return dict(metrics)

    @app.post("/diagnose")
    def diagnose(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            request = str(payload.get("request", "analyze this signal"))
            raw_signal = payload["signal"]
            signal = SignalRecord(sample_rate_hz=float(raw_signal["sample_rate_hz"]),
                                  channels={str(k): list(map(float, v)) for k, v in raw_signal["channels"].items()},
                                  metadata=dict(raw_signal.get("metadata", {})))
            task, report = language_agent.ask(request, signal=signal)
        except (KeyError, TypeError, ValueError) as exc:
            metrics["invalid_requests"] += 1
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        metrics["requests_total"] += 1
        metrics["escalations_total"] += int(report.fault_family == "uncertain")
        metrics[f"fault_{report.fault_family}"] += 1
        return {"task": task.__dict__, "report": report.to_dict()}

    return app
