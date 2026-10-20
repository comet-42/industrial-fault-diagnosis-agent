"""Run the CSV quickstart without private model or MATLAB dependencies."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from industrial_fault_agent import FaultDiagnosisAgent, SignalRecord


CHANNELS = ("i_a", "i_b", "i_c")


def load_csv(path: Path) -> SignalRecord:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(channel not in rows[0] for channel in CHANNELS):
        raise ValueError(f"CSV must contain columns: {', '.join(CHANNELS)}")
    channels = {channel: [float(row[channel]) for row in rows] for channel in CHANNELS}
    return SignalRecord(sample_rate_hz=2500.0, channels=channels,
                        metadata={"source": "reference-csv", "path": path.name})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the industrial fault Agent quickstart")
    parser.add_argument("--input", type=Path, required=True, help="CSV with i_a,i_b,i_c columns")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    report = FaultDiagnosisAgent().diagnose(load_csv(args.input))
    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
