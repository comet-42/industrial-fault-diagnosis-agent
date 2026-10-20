import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_manifest import evaluate_manifest


class ManifestEvaluationTests(unittest.TestCase):
    def test_manifest_reports_macro_f1_and_latency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            signal_path = root / "window.csv"
            with signal_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["i_a", "i_b", "i_c"])
                writer.writeheader()
                for _ in range(500):
                    writer.writerow({"i_a": 8.0, "i_b": 0.01, "i_c": 8.0})
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["path", "label", "sample_rate_hz"])
                writer.writeheader()
                writer.writerow({"path": str(signal_path), "label": "phase_open", "sample_rate_hz": 2500})
            report = evaluate_manifest(manifest)
            self.assertEqual(report["accuracy"], 1.0)
            self.assertEqual(report["macro_f1"], 1.0)
            self.assertIn("p95_latency_ms", report)


if __name__ == "__main__":
    unittest.main()
