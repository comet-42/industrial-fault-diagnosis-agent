"""Interactively configure the natural-language model for this project.

The API key is entered through ``getpass`` and written only to the local,
Git-ignored ``.env`` file. This script never sends the key over the network.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

# Allow the script to run directly from a fresh checkout before editable
# installation. The package still owns all configuration logic.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from industrial_fault_agent.config import write_llm_env


MODELS = {
    "1": ("deepseek-chat", "DeepSeek Chat（速度优先，推荐日常诊断请求）"),
    "2": ("deepseek-reasoner", "DeepSeek Reasoner（复杂任务推理优先）"),
}


def choose_model(initial: str | None = None) -> str:
    if initial:
        valid = {value[0] for value in MODELS.values()}
        if initial not in valid:
            raise ValueError(f"unsupported model {initial!r}; choose from {', '.join(sorted(valid))}")
        return initial
    print("请选择自然语言模型：")
    for key, (_, label) in MODELS.items():
        print(f"  {key}. {label}")
    choice = input("模型编号 [1]: ").strip() or "1"
    if choice not in MODELS:
        raise ValueError("请输入 1 或 2")
    return MODELS[choice][0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Configure DeepSeek for the industrial diagnosis Agent")
    parser.add_argument("--model", choices=[value[0] for value in MODELS.values()],
                        help="skip model selection prompt")
    parser.add_argument("--env-file", type=Path, default=Path.cwd() / ".env",
                        help="local env file path (default: project .env)")
    args = parser.parse_args()
    try:
        model = choose_model(args.model)
        api_key = getpass.getpass("请输入 DeepSeek API Key（输入不会显示）：").strip()
        if not api_key:
            raise ValueError("API Key 不能为空")
        path = write_llm_env(args.env_file, model=model, api_key=api_key)
    except (EOFError, KeyboardInterrupt):
        print("\n已取消配置。", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"配置失败：{exc}", file=sys.stderr)
        return 2
    print(f"已保存本地模型配置：{path}")
    print("之后可运行：python -m industrial_fault_agent.cli ask --parser deepseek --request \"分析 data/demo/phase_open_window.csv\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
