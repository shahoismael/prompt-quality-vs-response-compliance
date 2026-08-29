function parsed = safe_json_decode(raw_text)
    if isempty(raw_text), parsed = []; return; end
    
    % 1. Extract the outermost array or object
    start_arr = strfind(raw_text, '['); end_arr = strfind(raw_text, ']');
    if ~isempty(start_arr) && ~isempty(end_arr)
        json_str = raw_text(start_arr(1):end_arr(end));
    else
        start_obj = strfind(raw_text, '{'); end_obj = strfind(raw_text, '}');
        if ~isempty(start_obj) && ~isempty(end_obj)
            json_str = raw_text(start_obj(1):end_obj(end));
        else
            parsed = []; return;
        end
    end
    
    % 2. Clean up common LLM JSON errors
    % Remove trailing commas before closing brackets/braces
    json_str = regexprep(json_str, ',\s*]', ']');
    json_str = regexprep(json_str, ',\s*}', '}');
    
    % Fix unquoted keys (e.g., {score: 4} -> {"score": 4})
    json_str = regexprep(json_str, '([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', '$1"$2"$3');
    
    % 3. Try to decode
    try
        parsed = jsondecode(json_str);
    catch ME
        fprintf('=== JSON DECODE FAILED ===\n');
        fprintf('Raw text extracted:\n%s\n', json_str);
        fprintf('Error: %s\n', ME.message);
        fprintf('==========================\n');
        parsed = [];
    end
end