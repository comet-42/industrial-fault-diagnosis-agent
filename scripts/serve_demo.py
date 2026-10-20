"""Start the optional HTTP demonstration server."""

from __future__ import annotations

import argparse
import os

from industrial_fault_agent.config import load_local_env
from industrial_fault_agent.natural_language import DeepSeekTaskParser, RuleBasedTaskParser


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the industrial fault diagnosis Agent")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    load_local_env()
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install the API extra first: pip install -e '.[api]'") from exc
    from industrial_fault_agent.api import create_app
    parser_impl = RuleBasedTaskParser()
    if os.getenv("FAULT_AGENT_PARSER", "rules") == "deepseek":
        parser_impl = DeepSeekTaskParser(model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    uvicorn.run(create_app(parser=parser_impl), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
