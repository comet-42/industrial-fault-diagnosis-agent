import unittest

try:
    from fastapi.testclient import TestClient
    from industrial_fault_agent.api import create_app
except ImportError:  # pragma: no cover - optional API dependency
    TestClient = None
    create_app = None


@unittest.skipIf(TestClient is None, "install the optional API dependencies")
class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app())

    def test_health_and_metrics(self):
        self.assertEqual(self.client.get("/health").json()["status"], "ok")
        self.assertEqual(self.client.get("/metrics").json(), {})

    def test_diagnosis_endpoint_returns_trace_and_updates_metrics(self):
        signal = {
            "sample_rate_hz": 2500,
            "channels": {"i_a": [8.0] * 500, "i_b": [0.01] * 500, "i_c": [8.0] * 500},
        }
        response = self.client.post("/diagnose", json={"request": "diagnose this signal", "signal": signal})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["report"]["fault_family"], "phase_open")
        self.assertTrue(payload["report"]["decision_trace"])
        self.assertTrue(all("duration_ms" in item for item in payload["report"]["tool_trace"]))
        metrics = self.client.get("/metrics").json()
        self.assertEqual(metrics["requests_total"], 1)

    def test_invalid_payload_returns_422_and_is_counted(self):
        response = self.client.post("/diagnose", json={"signal": {}})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.client.get("/metrics").json()["invalid_requests"], 1)


if __name__ == "__main__":
    unittest.main()
