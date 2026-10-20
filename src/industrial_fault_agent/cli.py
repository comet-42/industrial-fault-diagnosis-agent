from __future__ import annotations

import argparse
import json

from .agent import FaultDiagnosisAgent
from .synthetic import FAULTS, generate_signal


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the industrial fault diagnosis agent")
    parser.add_argument("--fault", choices=FAULTS, default="phase_open", help="reference synthetic condition")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    report = FaultDiagnosisAgent().diagnose(generate_signal(args.fault, seed=args.seed))
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
