function predict_pa_jc_hmt_tcn(windowCsv, modelPath, outputJson)
%PREDICT_PA_JC_HMT_TCN Run the frozen PA-JC-HMT-TCN artifact.
arguments
    windowCsv (1,1) string
    modelPath (1,1) string
    outputJson (1,1) string
end
window = readmatrix(windowCsv);
assert(isequal(size(window), [15 500]), ...
    'Expected a standardized 15x500 CSV window, got %s.', mat2str(size(window)));
assert(all(isfinite(window), 'all'), 'Input window contains non-finite values.');
loaded = load(modelPath, 'network', 'spec');
assert(isfield(loaded, 'network'), 'Model artifact has no network variable.');
network = loaded.network;
modelSpec = loaded.spec;
dlX = dlarray(single(reshape(window, 15, 500, 1)), 'CTB');
[faultScores, moduleScores, rawSeverity] = predict(network, dlX, ...
    'Outputs', {'fault_softmax', 'module_softmax', 'severity_sigmoid'});
faultScores = double(gather(extractdata(faultScores)));
moduleScores = double(gather(extractdata(moduleScores)));
rawSeverity = double(gather(extractdata(rawSeverity)));
faultScores = reshape(faultScores, 1, []);
moduleScores = reshape(moduleScores, 1, []);
rawSeverity = rawSeverity(1);
[confidence, faultIndex] = max(faultScores);
[~, moduleIndex] = max(moduleScores);
faultNames = ["Normal", "OpenCircuit", "GroundUnbalance", "Interturn", ...
    "Demagnetization", "SensorBias", "SensorNoise"];
moduleNames = ["CommonOrNone", "Module1", "Module2", "Module3"];
faultType = faultIndex - 1;
if faultType == 0 || faultType == 4, module = 0; else, module = moduleIndex; end
if faultType == 0, severity = 0; else, severity = rawSeverity; end
result = struct('fault_type',faultType,'fault_name',char(faultNames(faultIndex)), ...
    'module',module,'module_name',char(moduleNames(module+1)), ...
    'severity',severity,'raw_severity',rawSeverity,'confidence',confidence, ...
    'fault_scores',faultScores,'module_scores_1_to_3',moduleScores, ...
    'model_file',char(modelPath));
if isfield(modelSpec, 'model_name'), result.model_name = char(string(modelSpec.model_name));
else, result.model_name = 'PA-JC-HMT-TCN'; end
fid = fopen(outputJson, 'w');
assert(fid > 0, 'Could not open output JSON: %s', outputJson);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid, jsonencode(result), 'char');
end
