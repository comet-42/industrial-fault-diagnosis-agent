"""Bridge from Python to the frozen MATLAB PA-JC-HMT-TCN artifact."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np


class MatlabPAJCTCNAdapter:
    """Run the real dlnetwork through MATLAB's supported batch interface."""

    def __init__(self, model_path: str, *, matlab_executable: Optional[str] = None,
                 bridge_path: Optional[str] = None, timeout_s: int = 120):
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        self.matlab_executable = matlab_executable or os.environ.get("MATLAB_EXECUTABLE", "matlab")
        self.bridge_path = Path(bridge_path or Path(__file__).resolve().parents[2] / "matlab" / "predict_pa_jc_hmt_tcn.m").resolve()
        if not self.bridge_path.is_file():
            raise FileNotFoundError(self.bridge_path)
        self.timeout_s = timeout_s

    def predict(self, window: np.ndarray) -> dict:
        value = np.asarray(window, dtype=np.float32)
        if value.shape != (15, 500):
            raise ValueError(f"PA-JC-HMT-TCN expects a standardized 15x500 window, got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError("PA-JC-HMT-TCN input contains non-finite values")
        with tempfile.TemporaryDirectory(prefix="pa_jc_agent_") as directory:
            root = Path(directory)
            csv_path, json_path = root / "window.csv", root / "result.json"
            np.savetxt(csv_path, value, delimiter=",")
            command = (f"addpath('{_matlab_quote(str(self.bridge_path.parent))}'); "
                       f"predict_pa_jc_hmt_tcn('{_matlab_quote(str(csv_path))}',"
                       f"'{_matlab_quote(str(self.model_path))}','{_matlab_quote(str(json_path))}');")
            completed = subprocess.run([self.matlab_executable, "-batch", command],
                                       capture_output=True, text=True, timeout=self.timeout_s)
            if completed.returncode != 0:
                raise RuntimeError("MATLAB PA-JC inference failed: " + (completed.stderr or completed.stdout)[-4000:])
            if not json_path.is_file():
                raise RuntimeError("MATLAB completed without writing a result JSON")
            result = json.loads(json_path.read_text(encoding="utf-8"))
        result["adapter"] = "matlab-pa-jc-hmt-tcn"
        return result


def _matlab_quote(value: str) -> str:
    return value.replace("'", "''")
