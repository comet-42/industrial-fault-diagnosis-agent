# Architecture and operational walkthrough

## Agent state

The workflow state consists of the signal schema check, engineered features, model prediction, retrieved engineering interpretation, and validation result. Each tool returns both a typed result and a `ToolTrace`, which makes the run inspectable.

## Reliability decisions

- The input is checked for finite values, consistent length, and a minimum window size.
- A model label is not exposed as a diagnosis until the validation tool approves it.
- Every report cites the signal inspection, feature values, model output, and knowledge entry as evidence.
- Uncertainty is represented explicitly as `fault_family="uncertain"` with an escalation action.
- The deterministic path is fully offline and reproducible. An LLM can propose actions later, but it cannot directly write a report or bypass the quality gate.

## Connecting the previous PMSM work

The intended production adapter is the existing PMSM TCN: concatenate the same 15-channel window representation, load the frozen artifact, and map its seven-class/multi-task output to the `run_diagnostic_model` contract. The Agent then adds the missing product layer around that model: tool orchestration, evidence, domain retrieval, refusal, and an evaluation surface.

## Suggested operational check

1. Run `python -m industrial_fault_agent.cli --fault phase_open --pretty`.
2. Verify that the report contains five tool calls and evidence rather than only a class label.
3. Change one input sample to `NaN` and verify that the Agent returns `uncertain` and escalates.
4. Replace `run_diagnostic_model` with an ONNX Runtime adapter without changing the orchestration or report schema.
