# Agent design

This repository uses a goal-driven, tool-using Agent rather than a fixed function pipeline.

```text
AgentState -> Planner -> ToolExecutor -> Observation -> AgentState
```

`AgentState` stores the task goal, observations, uncertainty, tool traces, decision traces, and status. `ToolSpec` describes a tool's precondition, utility score, and handler. `GoalDirectedPlanner` evaluates all currently applicable tools and selects the highest-scoring action. The loop stops when validation succeeds, no tool can advance the task, a tool fails, or the action budget is exhausted.

The policy is deterministic by design. This provides reproducible industrial behavior while preserving the same interface for a future LLM planner. An LLM may propose a tool action, but the executor should still enforce tool schemas, preconditions, timeouts, and validation gates.

The important distinction from a fixed script is the decision path. A normal request does not run targeted fault analysis. A sensor-bias or interturn candidate activates a different evidence tool. An ambiguous prediction activates an uncertainty gate. The decision trace records the candidate tools and selected action at each step, so the path can be inspected after the run.

The current policy is intentionally small and safe. Production expansion should add bounded evidence collection, adjacent-window acquisition, model fallback, human approval nodes, persistent task state, and metrics for tool success, escalation, latency, and calibration.
