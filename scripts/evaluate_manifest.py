"""Evaluate labeled, non-public signal windows without copying them into Git.

Manifest columns: ``path,label,sample_rate_hz``. CSV files use the public
three-column contract ``i_a,i_b,i_c``. The report includes accuracy,
per-class counts, escalation rate, and average latency.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from industrial_fault_agent import FaultDiagnosisAgent
from industrial_fault_agent.natural_language import load_signal_csv


def evaluate_manifest(path: Path) -> dict:
    agent = FaultDiagnosisAgent()
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(key not in rows[0] for key in ("path", "label")):
        raise ValueError("manifest must contain path,label columns")
    cases, confusion = [], defaultdict(Counter)
    for row in rows:
        signal = load_signal_csv(Path(row["path"]), sample_rate_hz=float(row.get("sample_rate_hz") or 2500))
        started = time.perf_counter()
        report = agent.diagnose(signal)
        latency = (time.perf_counter() - started) * 1000
        expected = row["label"]
        confusion[expected][report.fault_family] += 1
        cases.append({"path": row["path"], "expected": expected, "predicted": report.fault_family,
                      "correct": report.fault_family == expected, "escalated": report.fault_family == "uncertain",
                      "confidence": report.confidence, "latency_ms": round(latency, 3)})
    labels = sorted({case["expected"] for case in cases} | {case["predicted"] for case in cases})
    per_class = {}
    f1s = []
    for label in labels:
        tp = confusion[label][label]
        actual = sum(confusion[label].values())
        predicted = sum(confusion[other][label] for other in labels)
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": actual}
        if actual:
            f1s.append(f1)
    latencies = sorted(case["latency_ms"] for case in cases)
    p95_index = min(len(latencies) - 1, max(0, int(0.95 * len(latencies) + 0.999) - 1))
    return {"n_cases": len(cases), "accuracy": sum(c["correct"] for c in cases) / len(cases),
            "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
            "escalation_rate": sum(c["escalated"] for c in cases) / len(cases),
            "avg_latency_ms": sum(latencies) / len(latencies), "p95_latency_ms": latencies[p95_index],
            "per_class": per_class, "confusion": {key: dict(value) for key, value in confusion.items()}, "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Agent on a labeled signal manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate_manifest(args.manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
