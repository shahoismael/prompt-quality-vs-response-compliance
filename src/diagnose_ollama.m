%% Diagnostic: run this and paste the FULL output
endpoint = 'http://127.0.0.1:11434/api/chat';

% Using raw JSON string to avoid MATLAB struct-to-JSON array issues
json_str = '{"model": "qwen2.5:7b", "stream": false, "messages": [{"role": "user", "content": "Say hello in one sentence."}]}';

opts = weboptions('RequestMethod', 'post', 'ContentType', 'json', ...
    'MediaType', 'application/json', 'Timeout', 60);

fprintf('--- Testing connection to Ollama (qwen2.5:7b) ---\n');
try
    res = webwrite(endpoint, json_str, opts);
    
    if ischar(res) || isstring(res)
        res = jsondecode(res);
    end
    
    fprintf('SUCCESS.\n');
    fprintf('Model response: %s\n', res.message.content);
catch ME
    fprintf('FAILED.\n');
    fprintf('identifier: %s\n', ME.identifier);
    fprintf('message: %s\n', ME.message);
    fprintf('\nLikely cause: Ollama app is not running, or "ollama serve" is not active.\n');
end