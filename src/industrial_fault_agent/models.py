from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SignalRecord:
    """A compact PMSM window. Channels are three phase currents by default."""

    sample_rate_hz: float
    channels: Dict[str, List[float]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return len(next(iter(self.channels.values()))) if self.channels else 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskRequest:
    """Normalized intent extracted from a natural-language request.

    The diagnosis runtime consumes this typed object instead of interpreting
    free text inside tools.  This keeps language parsing replaceable while
    making safety-relevant requirements auditable in the report.
    """

    intent: str = "diagnose_signal"
    input_path: Optional[str] = None
    focus_fault: Optional[str] = None
    confidence_threshold: float = 0.60
    require_evidence: bool = False
    output_format: str = "json"
    original_request: str = ""

    def __post_init__(self) -> None:
        self.confidence_threshold = float(self.confidence_threshold)
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.output_format not in {"json", "text"}:
            raise ValueError("output_format must be 'json' or 'text'")


@dataclass
class ToolTrace:
    tool: str
    purpose: str
    input_summary: Dict[str, Any]
    output_summary: Dict[str, Any]
    status: str = "ok"


@dataclass
class AgentDecision:
    """A planner decision recorded before each tool invocation."""

    step: int
    chosen_tool: str
    rationale: str
    candidate_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class Evidence:
    source: str
    claim: str
    value: Any
    confidence: float


@dataclass
class DiagnosisReport:
    diagnosis: str
    fault_family: str
    severity: float
    confidence: float
    recommended_action: str
    evidence: List[Evidence] = field(default_factory=list)
    tool_trace: List[ToolTrace] = field(default_factory=list)
    decision_trace: List[AgentDecision] = field(default_factory=list)
    uncertainty: List[str] = field(default_factory=list)
    run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
