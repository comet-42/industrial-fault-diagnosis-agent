# Two-minute interview walkthrough

## Demo sequence

1. Run `python scripts/run_demo.py --input data/demo/phase_open_window.csv`.
2. Show the report's `fault_family`, confidence, recommendation, evidence, and five-step tool trace.
3. Open `src/industrial_fault_agent/agent.py` and explain that the report is only issued after inspection, feature extraction, model inference, knowledge retrieval, and validation.
4. Show `matlab/predict_pa_jc_hmt_tcn.m` and explain the real three-head dlnetwork call and the explicit `15x500` contract.
5. Run the test suite and point out the invalid-input refusal test.

## Suggested explanation

"I already had a PMSM fault model, but a classifier alone did not demonstrate Agent engineering. I wrapped the frozen multi-task model in a typed workflow: inspect the input, run the model, retrieve an engineering interpretation, and validate before writing a report. Each step leaves a trace and the system escalates malformed or low-confidence cases. The public demo is synthetic, while the MATLAB adapter calls the actual PA-JC-HMT-TCN artifact when it is available locally."

## Questions to prepare for

- Why not let an LLM directly diagnose the signal? Because the model and quality gates should remain deterministic and auditable.
- Why is the model not uploaded? The artifact and research data are private; the bridge and contract are reproducible without exposing them.
- What would you improve? Add grouped real-machine validation, calibrated thresholds, an approved document retriever, and an optional LangGraph planner that cannot bypass the validation gate.
