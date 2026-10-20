# Model card: PA-JC-HMT-TCN Agent integration

## Intended use

The project demonstrates an evidence-grounded software Agent around the PA-JC-HMT-TCN multi-task model. Given a valid standardized window, it reports a seven-class fault type, module prediction, severity estimate, evidence, and an engineering action.

## Model and data boundary

The connected artifact is the frozen `TCN_V2` dlnetwork checkpoint. Its reported confirmation evaluation used system-level dq/abc proxy-fault simulation for a 159 kW three-module nine-phase PMSM. It is not real motor measurement data and is not a safety-certified protection controller.

## Input and output

- Input: standardized `15x500` window.
- Raw preprocessing: 10 kHz, 200 ms, non-overlapping 4-point block average.
- Channel order: `i1a,i1b,i1c,i2a,i2b,i2c,i3a,i3b,i3c,rd1,rq1,rd2,rq2,rd3,rq3`.
- Fault classes: Normal, OpenCircuit, GroundUnbalance, Interturn, Demagnetization, SensorBias, SensorNoise.
- Heads: fault type, module, severity.

## Confirmed evaluation record

The local v6.2 confirmation summary reports 84 independent runs, three proposed seeds, seven classes, and a run-level seven-class Macro-F1 of 0.9028 for the pressure joint TCN comparator. The summary also reports a positive confirmation decision. These figures are included as provenance for the research model and are not the accuracy of the synthetic public demo.

## Limitations

- Simulation-to-real transfer is unverified here.
- The Agent cannot recover incorrect normalization statistics or malformed windows.
- An engineering action is a recommendation, not an automatic control command.
- A qualified engineer must review deployment thresholds, drift monitoring, and escalation policy.

## Reproducibility

Keep the `.mat` checkpoint and private data outside Git. Record its SHA-256, MATLAB release, `spec` structure, preprocessing version, and seed when publishing a run. The public repository contains a MATLAB bridge, a Python adapter, synthetic CSVs, and contract tests.
