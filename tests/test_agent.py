import unittest
from unittest.mock import patch

from industrial_fault_agent.agent import FaultDiagnosisAgent
from industrial_fault_agent.evaluate import evaluate
from industrial_fault_agent.synthetic import generate_signal
from industrial_fault_agent.matlab_adapter import MatlabPAJCTCNAdapter


class AgentTests(unittest.TestCase):
    def test_all_public_demo_faults(self):
        result = evaluate()
        self.assertEqual(result["accuracy"], 1.0)
        self.assertGreaterEqual(result["avg_tool_calls"], 5.0)

    def test_report_has_evidence_and_trace(self):
        report = FaultDiagnosisAgent().diagnose(generate_signal("sensor_bias"))
        self.assertEqual(report.fault_family, "sensor_bias")
        self.assertGreaterEqual(len(report.evidence), 4)
        self.assertEqual([x.tool for x in report.tool_trace][-1], "validate_diagnosis")

    def test_planner_selects_targeted_tool_for_sensor_bias(self):
        report = FaultDiagnosisAgent().diagnose(generate_signal("sensor_bias"))
        self.assertIn("analyze_fault_evidence", [item.chosen_tool for item in report.decision_trace])
        self.assertGreater(len(report.decision_trace), 5)

    def test_normal_path_skips_targeted_fault_tool(self):
        report = FaultDiagnosisAgent().diagnose(generate_signal("normal"))
        chosen = [item.chosen_tool for item in report.decision_trace]
        self.assertNotIn("analyze_fault_evidence", chosen)
        self.assertEqual(chosen[-1], "validate_diagnosis")

    def test_low_confidence_replans_through_uncertainty_gate(self):
        def ambiguous_model(features):
            from industrial_fault_agent.models import ToolTrace
            return ({"label": "normal", "confidence": 0.52,
                     "class_probabilities": {"normal": 0.52, "sensor_bias": 0.48},
                     "model": "test-ambiguous"},
                    ToolTrace("run_diagnostic_model", "test", {}, {"confidence": 0.52}))

        with patch("industrial_fault_agent.agent.run_diagnostic_model", ambiguous_model):
            report = FaultDiagnosisAgent().diagnose(generate_signal("normal"))
        chosen = [item.chosen_tool for item in report.decision_trace]
        self.assertIn("assess_uncertainty", chosen)
        self.assertEqual(report.fault_family, "uncertain")
        self.assertTrue(report.uncertainty)

    def test_invalid_signal_is_withheld(self):
        signal = generate_signal("normal")
        signal.channels["i_a"][0] = float("nan")
        report = FaultDiagnosisAgent().diagnose(signal)
        self.assertEqual(report.fault_family, "uncertain")
        self.assertTrue(report.uncertainty)

    def test_pa_jc_adapter_enforces_contract_without_running_matlab(self):
        adapter = object.__new__(MatlabPAJCTCNAdapter)
        with self.assertRaises(ValueError):
            adapter.predict([[0.0]])


if __name__ == "__main__":
    unittest.main()
