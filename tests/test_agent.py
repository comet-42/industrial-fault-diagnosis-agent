import unittest

from industrial_fault_agent.agent import FaultDiagnosisAgent
from industrial_fault_agent.evaluate import evaluate
from industrial_fault_agent.synthetic import generate_signal


class AgentTests(unittest.TestCase):
    def test_all_public_demo_faults(self):
        result = evaluate()
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["avg_tool_calls"], 5.0)

    def test_report_has_evidence_and_trace(self):
        report = FaultDiagnosisAgent().diagnose(generate_signal("sensor_bias"))
        self.assertEqual(report.fault_family, "sensor_bias")
        self.assertGreaterEqual(len(report.evidence), 4)
        self.assertEqual([x.tool for x in report.tool_trace][-1], "validate_diagnosis")

    def test_invalid_signal_is_withheld(self):
        signal = generate_signal("normal")
        signal.channels["i_a"][0] = float("nan")
        report = FaultDiagnosisAgent().diagnose(signal)
        self.assertEqual(report.fault_family, "uncertain")
        self.assertTrue(report.uncertainty)


if __name__ == "__main__":
    unittest.main()
