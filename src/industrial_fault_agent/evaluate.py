from __future__ import annotations

from typing import Dict

from .agent import FaultDiagnosisAgent
from .synthetic import FAULTS, generate_signal


def evaluate(seed: int = 100) -> Dict[str, object]:
    agent = FaultDiagnosisAgent()
    rows = []
    for i, label in enumerate(FAULTS):
        report = agent.diagnose(generate_signal(label, seed=seed + i))
        rows.append({"expected": label, "predicted": report.fault_family,
                     "correct": report.fault_family == label,
                     "confidence": report.confidence, "tool_calls": len(report.tool_trace)})
    accuracy = sum(row["correct"] for row in rows) / len(rows)
    return {"accuracy": accuracy, "n_cases": len(rows), "avg_tool_calls": sum(r["tool_calls"] for r in rows) / len(rows), "cases": rows}


if __name__ == "__main__":
    import json
    print(json.dumps(evaluate(), indent=2))
