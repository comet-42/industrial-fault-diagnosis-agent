# Operational walkthrough

## Quickstart validation

1. Run `python -m industrial_fault_agent.cli --fault phase_open --pretty`.
2. Verify that the report contains tool calls, planner decisions, evidence, a recommendation, and a run ID.
3. Change one input sample to `NaN` and verify that the Agent returns `uncertain` and escalates.
4. Run `python scripts/run_demo.py --input data/demo/phase_open_window.csv` to validate the CSV ingestion path.

## Production integration check

1. Provide a standardized `15x500` input with the documented channel order.
2. Configure a local PA-JC-HMT-TCN `.mat` artifact and MATLAB executable.
3. Run the MATLAB adapter and verify the model name, fault scores, module output, and severity output.
4. Store the structured report with its model version, preprocessing version, and run ID.

## Design rationale

The model and quality gates remain deterministic and auditable. A language-model planner may be added later for routing or explanation behind the same tool contract, but it cannot bypass schema validation, model inference, or escalation rules.
