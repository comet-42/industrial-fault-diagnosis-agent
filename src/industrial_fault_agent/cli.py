from __future__ import annotations

import argparse
import json
import os
import sys

from .agent import FaultDiagnosisAgent
from .natural_language import DeepSeekTaskParser, NaturalLanguageAgent, RuleBasedTaskParser
from .synthetic import FAULTS, generate_signal


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run the industrial fault diagnosis agent")
    subparsers = parser.add_subparsers(dest="command")
    ask = subparsers.add_parser("ask", help="diagnose a signal from a natural-language request")
    ask.add_argument("--request", required=True, help="natural-language task, optionally containing a CSV path")
    ask.add_argument("--input", help="CSV path; overrides a path mentioned in the request")
    ask.add_argument("--parser", choices=("rules", "deepseek"),
                     default=os.getenv("FAULT_AGENT_PARSER", "rules"),
                     help="natural-language parser (default: rules; set DEEPSEEK_API_KEY for deepseek)")
    ask.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                     help="DeepSeek model name when --parser deepseek is selected")
    ask.add_argument("--pretty", action="store_true")

    parser.add_argument("--fault", choices=FAULTS, default="phase_open", help="reference synthetic condition")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    if args.command == "ask":
        task_parser = RuleBasedTaskParser() if args.parser == "rules" else DeepSeekTaskParser(model=args.model)
        task, report = NaturalLanguageAgent(parser=task_parser).ask(args.request, input_path=args.input)
        payload = {"task": task.__dict__, "report": report.to_dict()}
        print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
        return
    report = FaultDiagnosisAgent().diagnose(generate_signal(args.fault, seed=args.seed))
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
