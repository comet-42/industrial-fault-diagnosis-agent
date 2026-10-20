import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from industrial_fault_agent.agent import AgentState, LLMPlanner, FaultDiagnosisAgent
from industrial_fault_agent.config import load_local_env, write_llm_env
from industrial_fault_agent.models import TaskRequest
from industrial_fault_agent.natural_language import DeepSeekTaskParser, NaturalLanguageAgent, RuleBasedTaskParser
from industrial_fault_agent.synthetic import generate_signal


class _FakeDeepSeek:
    class _Completions:
        @staticmethod
        def create(**kwargs):
            class Message:
                content = '{"input_path": null, "focus_fault": "phase_open", "confidence_threshold": 0.8, "require_evidence": true, "output_format": "json"}'
            class Choice:
                message = Message()
            class Response:
                choices = [Choice()]
            return Response()

    class _Chat:
        pass

    chat = _Chat()


_FakeDeepSeek.chat.completions = _FakeDeepSeek._Completions()


class NaturalLanguageTests(unittest.TestCase):
    def test_chinese_request_becomes_typed_task(self):
        task = RuleBasedTaskParser().parse(
            "分析 data/demo/phase_open_window.csv，如果置信度低于0.8就不要下结论并给出证据"
        )
        self.assertEqual(task.input_path, "data/demo/phase_open_window.csv")
        self.assertEqual(task.focus_fault, "phase_open")
        self.assertEqual(task.confidence_threshold, 0.8)
        self.assertTrue(task.require_evidence)

    def test_natural_language_request_changes_confidence_gate(self):
        task, report = NaturalLanguageAgent().ask(
            "分析这段电机数据，如果置信度低于0.8就不要下结论",
            signal=generate_signal("normal"),
        )
        self.assertEqual(task.confidence_threshold, 0.8)
        self.assertEqual(report.fault_family, "uncertain")
        self.assertTrue(report.uncertainty)

    def test_llm_planner_rejects_unregistered_tool(self):
        agent = FaultDiagnosisAgent()
        planner = LLMPlanner(lambda state, tools: {"tool": "delete_everything"}, tools=agent.planner.tools)
        with self.assertRaises(ValueError):
            planner.choose(AgentState(signal=generate_signal("normal"), task_request=TaskRequest()))

    def test_llm_planner_can_choose_available_registered_tool(self):
        agent = FaultDiagnosisAgent()
        planner = LLMPlanner(lambda state, tools: {"tool": tools[0]["name"], "rationale": "input quality must be checked first"}, tools=agent.planner.tools)
        choice = planner.choose(AgentState(signal=generate_signal("normal"), task_request=TaskRequest()))
        self.assertEqual(choice[0].name, "inspect_signal")
        self.assertEqual(planner.last_rationale, "input quality must be checked first")

    def test_deepseek_parser_validates_structured_response(self):
        parser = DeepSeekTaskParser(client=_FakeDeepSeek())
        task = parser.parse("分析这段电机数据，置信度低于0.8不要下结论")
        self.assertEqual(task.focus_fault, "phase_open")
        self.assertEqual(task.confidence_threshold, 0.8)
        self.assertTrue(task.require_evidence)

    def test_local_model_config_is_quoted_and_preserves_other_settings(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("ONNX_MODEL_PATH=data/private/model.onnx\n", encoding="utf-8")
            write_llm_env(path, model="deepseek-chat", api_key='sk-test#value"x')
            text = path.read_text(encoding="utf-8")
            self.assertIn("ONNX_MODEL_PATH=data/private/model.onnx", text)
            self.assertIn('DEEPSEEK_API_KEY="sk-test#value\\"x"', text)
            values = load_local_env(path)
            self.assertEqual(values["DEEPSEEK_MODEL"], "deepseek-chat")

    def test_tool_catalog_is_machine_readable_and_trace_has_latency(self):
        agent = FaultDiagnosisAgent()
        catalog = agent.planner.tool_catalog()
        self.assertTrue(all({"name", "description", "input_schema", "output_schema"} <= set(item) for item in catalog))
        report = agent.diagnose(generate_signal("sensor_bias"))
        self.assertTrue(all(item.duration_ms >= 0 for item in report.tool_trace))
        self.assertGreaterEqual(report.duration_ms, 0)
        self.assertTrue(report.decision_trace[0].available_tools)


if __name__ == "__main__":
    unittest.main()
