# Evaluation protocol

The included harness is a repeatable reference evaluation over five deterministic cases. It is useful for CI and integration checks; it is not a claim of field performance.

For the real PMSM model, keep the following protocol:

1. Split by complete operating-condition groups, machine runs, or source recordings.
2. Prevent overlapping windows from crossing a split.
3. Report macro-F1 for fault classes, module localization accuracy, severity regression error, calibration, median/p95 latency, tool-call count, and escalation rate.
4. Include corrupt, short, missing-channel, and out-of-distribution windows.
5. Store model version, feature schema version, and run ID with every report.
