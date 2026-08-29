function api = Ollama_API(model_name)
% Local Ollama client.
%
% FIX (v2): `temperature` is NOT a valid top-level field of Ollama's
% /api/chat request body. Ollama unmarshals the body into a Go struct and
% silently discards unknown top-level keys, so the previous version's
% "temperature": X was ignored on every call and every stage ran at the
% served model's default. Temperature must be nested inside "options".
%
% FIX (v2): "keep_alive" pins the model in memory between calls. Without it
% Ollama unloads the model after its idle timeout and the next call pays a
% full reload; that is the cause of the isolated multi-thousand-second
% latencies observed in the previous run.

    if nargin < 1 || isempty(model_name)
        model_name = 'qwen2.5:7b';
    end
    api.model_name = model_name;
    % 127.0.0.1, NOT localhost. On Windows, 'localhost' resolves to the IPv6
    % loopback ::1 first, while Ollama binds only the IPv4 loopback. MATLAB's
    % webwrite then fails with ConnectionRefused even though the server is up
    % and 'ollama serve' reports the port as already in use.
    api.endpoint = 'http://127.0.0.1:11434/api/chat';
    api.keep_alive = '30m';
    api.call = @(prompt, temp) call_ollama(api, prompt, temp);
    api.call_ensemble = @(prompt, temp, n) call_ollama_ensemble(api, prompt, temp, n);
end

function responses = call_ollama_ensemble(api, prompt, temp, n)
    if nargin < 4 || isempty(n), n = 3; end
    responses = cell(1, n);
    for k = 1:n
        responses{k} = call_ollama(api, prompt, temp);
    end
end

function resp = call_ollama(api, prompt, temp)
    if nargin < 3 || isempty(temp)
        temp = 0.7;
    end

    prompt_escaped = json_escape(prompt);

    % Temperature lives inside "options".
    %
    % DO NOT pin "seed" here. A fixed seed makes every sample in an ensemble
    % byte-identical, so sample variance is exactly zero on every item:
    % confidence C = 1 - var/V_max is always 1, nothing is ever routed to
    % review, and bias variance is always 0. That silently disables the
    % ensembling, the confidence router, and the stability metric at once.
    % Run-level reproducibility comes from rng(42) in run_simulation (item
    % sampling, typo injection, shuffling); the model's own stochasticity is
    % meant to be CHARACTERISED by the ensemble, not suppressed.
    jsonData = sprintf(['{"model": "%s", "stream": false, ' ...
        '"keep_alive": "%s", ' ...
        '"options": {"temperature": %f}, ' ...
        '"messages": [{"role": "user", "content": "%s"}]}'], ...
        api.model_name, api.keep_alive, temp, prompt_escaped);

    options = weboptions( ...
        'MediaType', 'application/json', ...
        'Timeout', 300 ...   % was 120; long generations on CPU exceeded it
    );

    resp = '';
    try
        res = webwrite(api.endpoint, jsonData, options);

        if ischar(res) || isstring(res)
            res = jsondecode(res);
        end

        if isstruct(res) && isfield(res, 'error')
            fprintf('=== Ollama API error: %s\n', res.error);
            return;
        end

        if isstruct(res) && isfield(res, 'message') && isfield(res.message, 'content')
            resp = res.message.content;
        else
            fprintf('=== Ollama API: unexpected response shape.\n');
        end
    catch ME
        fprintf('=== Ollama API FAILED\nidentifier: %s\nmessage: %s\n', ...
            ME.identifier, ME.message);
    end
end

function s = json_escape(s)
% Escape a MATLAB char row vector for embedding in a JSON string literal.
% Order matters: backslash first, then quotes, then control characters.
    s = strrep(s, '\', '\\');
    s = strrep(s, '"', '\"');
    s = strrep(s, char(13), '\r');
    s = strrep(s, char(10), '\n');
    s = strrep(s, char(9),  '\t');
    % strip remaining C0 control characters, which are illegal raw in JSON
    s(double(s) < 32) = ' ';
end
