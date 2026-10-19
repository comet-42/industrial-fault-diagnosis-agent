"""Optional adapter for a frozen ONNX classifier artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


class OnnxClassifier:
    """Loads a classifier with input [batch, channel, sample] and logits output.

    Keep private ONNX artifacts outside the Git repository. The label order is
    part of the model contract and must match the export used for training.
    """

    def __init__(self, model_path: str, labels: Sequence[str], providers=None):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("Install the optional adapter with: pip install -e '.[onnx]'") from exc
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        self.labels = tuple(labels)
        self.session = ort.InferenceSession(str(path), providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.model_version = f"{path.stem}:{path.stat().st_size}bytes"

    def predict(self, tensor: np.ndarray) -> dict:
        value = np.asarray(tensor, dtype=np.float32)
        if value.ndim == 2:
            value = value[None, ...]
        if value.ndim != 3 or value.shape[0] != 1:
            raise ValueError("expected [channels, samples] or [1, channels, samples]")
        logits = np.asarray(self.session.run(None, {self.input_name: value})[0])[0].astype(float)
        if len(logits) != len(self.labels):
            raise ValueError(f"model returned {len(logits)} scores for {len(self.labels)} labels")
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        index = int(probs.argmax())
        return {"label": self.labels[index], "confidence": float(probs[index]),
                "class_probabilities": dict(zip(self.labels, map(float, probs))),
                "model": self.model_version}
