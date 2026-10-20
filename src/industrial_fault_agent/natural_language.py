"""Natural-language entry points for the industrial diagnosis Agent.

The parser produces a small, validated :class:`TaskRequest`; it never gets to
call diagnostic tools directly.  This separation lets a local rule parser,
DeepSeek, another OpenAI-compatible endpoint, or an on-premise language model
share the same runtime and safety gates.
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Optional, Protocol

from .agent import FaultDiagnosisAgent
from .config import load_local_env
from .models import DiagnosisReport, SignalRecord, TaskRequest


class TaskParser(Protocol):
    def parse(self, request: str) -> TaskRequest:
        ...


_FAULT_ALIASES = {
    "normal": "normal", "正常": "normal",
    "phase_open": "phase_open", "open circuit": "phase_open", "open-circuit": "phase_open", "缺相": "phase_open", "开路": "phase_open",
    "interturn_short": "interturn_short", "inter-turn": "interturn_short", "turn short": "interturn_short", "匝间短路": "interturn_short", "匝间": "interturn_short",
    "demagnetization": "demagnetization", "退磁": "demagnetization",
    "sensor_bias": "sensor_bias", "sensor bias": "sensor_bias", "传感器偏置": "sensor_bias", "传感器偏差": "sensor_bias",
    "sensor_noise": "sensor_noise", "sensor noise": "sensor_noise", "传感器噪声": "sensor_noise",
    "ground_unbalance": "ground_unbalance", "ground unbalance": "ground_unbalance", "接地不平衡": "ground_unbalance",
}


class RuleBasedTaskParser:
    """Offline parser for common Chinese and English operator requests."""

    def parse(self, request: str) -> TaskRequest:
        if not isinstance(request, str) or not request.strip():
            raise ValueError("natural-language request must be a non-empty string")
        text = request.strip()
        return TaskRequest(
            intent="diagnose_signal",
            input_path=self._input_path(text),
            focus_fault=self._focus_fault(text),
            confidence_threshold=self._threshold(text),
            require_evidence=bool(re.search(r"证据|依据|验证|支撑|evidence|validate|verify", text, re.I)),
            output_format="text" if re.search(r"纯文本|文字报告|text report", text, re.I) else "json",
            original_request=text,
        )

    @staticmethod
    def _input_path(text: str) -> Optional[str]:
        quoted = re.search(r"[\"']([^\"']+\.(?:csv|json|npy))[\"']", text, re.I)
        if quoted:
            return quoted.group(1)
        bare = re.search(r"(?<!\S)([^\s，。；;,]+\.(?:csv|json|npy))", text, re.I)
        return bare.group(1) if bare else None

    @staticmethod
    def _threshold(text: str) -> float:
        patterns = (
            r"(?:低于|小于|不低于|至少|threshold\s*(?:of|=)?|confidence\s*(?:of|=)?)\s*(\d+(?:\.\d+)?%?)",
        )
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                raw = match.group(1)
                value = float(raw.rstrip("%"))
                return value / 100.0 if raw.endswith("%") or value > 1 else value
        return 0.60

    @staticmethod
    def _focus_fault(text: str) -> Optional[str]:
        lowered = text.casefold()
        for alias, label in sorted(_FAULT_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
            if alias == "normal":
                # Avoid treating ``normal`` in ``normal_window.csv`` as an
                # operator request for the normal fault class.
                if re.search(r"(?<![_\w])normal(?![_\w])", lowered):
                    return label
                continue
            if alias.casefold() in lowered:
                return label
        return None


class OpenAITaskParser:
    """Optional structured parser backed by an OpenAI-compatible SDK client.

    The returned JSON is validated by :class:`TaskRequest` and unknown fields
    are ignored.  Install the optional dependency with ``pip install -e '.[llm]'``;
    callers can inject a fake ``client`` for offline tests.
    """

    def __init__(self, client: Any = None, *, model: str = "gpt-5-mini",
                 api_key: Optional[str] = None, base_url: Optional[str] = None):
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise RuntimeError("OpenAI parser requires the optional 'llm' dependency") from exc
            client = OpenAI(api_key=api_key, base_url=base_url)
        self.client = client
        self.model = model
        self.response_format = {"type": "json_schema", "json_schema": {"name": "task_request", "strict": True, "schema": self._schema()}}

    @staticmethod
    def _schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input_path": {"type": ["string", "null"]},
                "focus_fault": {"type": ["string", "null"]},
                "confidence_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                "require_evidence": {"type": "boolean"},
                "output_format": {"type": "string", "enum": ["json", "text"]},
            },
            "required": ["input_path", "focus_fault", "confidence_threshold", "require_evidence", "output_format"],
            "additionalProperties": False,
        }

    def parse(self, request: str) -> TaskRequest:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": "Extract a diagnosis task. Return only the requested JSON schema. Do not invent paths or fault labels."},
                {"role": "user", "content": request},
            ],
            response_format=self.response_format,
        )
        content = response.choices[0].message.content
        values = json.loads(content)
        return self._task_from_values(values, request)

    @staticmethod
    def _task_from_values(values: dict[str, Any], request: str) -> TaskRequest:
        allowed_faults = set(_FAULT_ALIASES.values())
        focus = values.get("focus_fault")
        if focus and focus not in allowed_faults:
            focus = None
        return TaskRequest(
            input_path=values.get("input_path"), focus_fault=focus,
            confidence_threshold=values.get("confidence_threshold", 0.60),
            require_evidence=bool(values.get("require_evidence", False)),
            output_format=values.get("output_format", "json"), original_request=request,
        )


class DeepSeekTaskParser(OpenAITaskParser):
    """Natural-language parser using DeepSeek's OpenAI-compatible API.

    DeepSeek's JSON-object mode is used instead of assuming provider-specific
    JSON-schema support.  The response is still validated locally before it
    can affect the diagnosis Agent.
    """

    def __init__(self, client: Any = None, *, model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 base_url: str = "https://api.deepseek.com"):
        load_local_env()
        key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if client is None and not key:
            raise RuntimeError("DeepSeek parser requires DEEPSEEK_API_KEY")
        super().__init__(client, model=model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                         api_key=key, base_url=base_url)
        self.response_format = {"type": "json_object"}


def load_signal_csv(path: str | Path, *, sample_rate_hz: float = 2500.0) -> SignalRecord:
    """Load the public three-phase CSV contract used by the quickstart."""
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    channels = ("i_a", "i_b", "i_c")
    if not rows or any(channel not in rows[0] for channel in channels):
        raise ValueError("CSV must contain columns: i_a, i_b, i_c")
    return SignalRecord(sample_rate_hz=sample_rate_hz,
                        channels={channel: [float(row[channel]) for row in rows] for channel in channels},
                        metadata={"source": "csv", "path": str(csv_path)})


class NaturalLanguageAgent:
    """Bridge natural-language requests into the existing diagnosis runtime."""

    def __init__(self, agent: Optional[FaultDiagnosisAgent] = None,
                 parser: Optional[TaskParser] = None):
        self.agent = agent or FaultDiagnosisAgent()
        self.parser = parser or RuleBasedTaskParser()

    def ask(self, request: str, *, signal: Optional[SignalRecord] = None,
             input_path: Optional[str | Path] = None) -> tuple[TaskRequest, DiagnosisReport]:
        task = self.parser.parse(request)
        selected_path = str(input_path) if input_path is not None else task.input_path
        if signal is None:
            if not selected_path:
                raise ValueError("provide a SignalRecord or a CSV path in the request/--input option")
            signal = load_signal_csv(selected_path)
        if input_path is not None:
            task.input_path = str(input_path)
        return task, self.agent.diagnose(signal, task_request=task)
