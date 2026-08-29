% --- Qwen 2.5 Readiness Test ---
fprintf('Initializing API wrapper...\n');
api = Ollama_API('qwen2.5:7b');

% Test 1: Basic Text Generation
fprintf('\n--- Test 1: Basic Text Generation ---\n');
prompt1 = 'What is 2 + 2? Reply with only the number.';
resp1 = api.call(prompt1, 0.1);
fprintf('Prompt: %s\n', prompt1);
fprintf('Response: %s\n', resp1);

if isempty(resp1)
    error('Test 1 failed: No response received. Check if Ollama is running.');
else
    fprintf('Test 1 PASSED.\n');
end

% Test 2: JSON Generation (Critical for ExtractorAgent & SemanticScorer)
fprintf('\n--- Test 2: JSON Extraction ---\n');
prompt2 = sprintf(['You are an expert educator. Analyze this student prompt and extract ALL constraints:\n\n' ...
    'Prompt: "Write a 100 word essay about the ocean in bullet points."\n\n' ...
    'Return a JSON array with objects containing:\n' ...
    '- "type": "strict" or "loose"\n' ...
    '- "description": what the constraint requires\n' ...
    '- "verifiable": true for strict, false for loose\n\n' ...
    'Return ONLY a JSON array. Example: [{"type":"strict","description":"Use bullet points","verifiable":true}]']);

resp2 = api.call(prompt2, 0.1);
fprintf('Raw JSON Response:\n%s\n', resp2);

parsed = safe_json_decode(resp2);
if ~isempty(parsed) && isstruct(parsed)
    fprintf('Test 2 PASSED. Successfully parsed %d constraints.\n', length(parsed));
else
    error('Test 2 failed: JSON parsing failed. The model might be wrapping JSON in markdown.');
end

fprintf('\n========================================\n');
fprintf('All tests passed! Qwen 2.5 is ready for the simulation pipeline.');
fprintf('\n========================================\n');