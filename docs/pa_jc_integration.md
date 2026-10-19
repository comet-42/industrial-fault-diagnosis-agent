# PA-JC-HMT-TCN integration

The connected artifact is the local development checkpoint:

```text
results/method_v3/critical_baselines_seed56020/TCN_V2/seed_51001/development_network.mat
```

The artifact contains `network`, `spec`, and validation metadata. Its network has 38 layers and three output heads. The frozen model was trained on system-level dq/abc proxy-fault simulation, not real motor measurements.

## Input contract

The model consumes one **standardized** 15-channel, 500-sample window:

```text
raw: 10 kHz x 200 ms = 2000 samples
preprocess: non-overlapping 4-point block average
model: 2.5 kHz x 500 samples
channels: i1a,i1b,i1c,i2a,i2b,i2c,i3a,i3b,i3c,rd1,rq1,rd2,rq2,rd3,rq3
```

The normalization statistics must be the original core training split statistics. The Agent deliberately does not guess or recompute them. Use the same preprocessing/export pipeline that generated the frozen `XTrain`/`XValidation` arrays before passing a window to the adapter.

## Output contract

The MATLAB bridge returns the fault class and probability vector, module prediction, severity, raw severity, model name, and artifact path. Fault labels are `Normal`, `OpenCircuit`, `GroundUnbalance`, `Interturn`, `Demagnetization`, `SensorBias`, and `SensorNoise`.

## Verification performed

The bridge was run against `seed_51001/development_network.mat` and a real `15x500` validation window from the development dataset. It returned `Normal` with confidence `0.7384`, module `CommonOrNone`, and severity `0`. The Python Agent then converted that output into an evidence-grounded report with a two-step trace: frozen model inference followed by engineering knowledge retrieval.

## Privacy and reproducibility

Model artifacts, raw datasets, and private experiment paths are not committed. Configure the local artifact path explicitly, record its SHA-256 in an experiment log, and include the MATLAB release and model `spec` when reporting results.
