# Architecture and operational walkthrough

## Agent state

The Agent state contains the task goal, observations, uncertainty, tool traces, decision traces, and lifecycle status. A `GoalDirectedPlanner` evaluates a registry of tools, applies each tool's precondition and utility score, executes the highest-scoring applicable action, and replans from the new observation.

## Reliability decisions

- The input is checked for finite values, consistent length, and a minimum window size.
- A model label is not exposed as a diagnosis until the validation tool approves it.
- Every report cites the signal inspection, feature values, model output, and knowledge entry as evidence.
- Uncertainty is represented explicitly as `fault_family="uncertain"` with an escalation action.
- The policy is deterministic and fully offline, so it is reproducible. A future LLM planner can propose actions through the same tool contract, but it cannot directly write a report or bypass the quality gate.

## Dynamic paths

The planner does not call every tool on every request. Normal and open-phase cases use the core inspection, feature, model, knowledge, and validation path. Sensor-bias and interturn predictions activate `analyze_fault_evidence`; low-confidence predictions activate `assess_uncertainty`. A tool failure or exhausted action budget terminates with escalation.

## Connecting the previous PMSM work

The intended production adapter is the existing PMSM TCN: concatenate the same 15-channel window representation, load the frozen artifact, and map its seven-class/multi-task output to the `run_diagnostic_model` contract. The Agent then adds the missing product layer around that model: tool orchestration, evidence, domain retrieval, refusal, and an evaluation surface.

## Suggested operational check

1. Run `python -m industrial_fault_agent.cli --fault phase_open --pretty`.
2. Verify that the report contains five tool calls and evidence rather than only a class label.
3. Change one input sample to `NaN` and verify that the Agent returns `uncertain` and escalates.
4. Replace `run_diagnostic_model` with an ONNX Runtime adapter without changing the orchestration or report schema.
