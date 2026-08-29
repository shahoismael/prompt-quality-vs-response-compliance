function extracted = ExtractorAgent(prompt, api)
    % Escaping is now handled centrally inside Ollama_API.m to prevent 
    % double-escaping of quotes which breaks the JSON payload.
    
    extraction_prompt = sprintf(['You are an expert educator. Analyze this student prompt and extract ALL constraints:\n\n' ...
        'Prompt: "%s"\n\n' ...
        'Return a JSON array with objects containing:\n' ...
        '- "type": "strict" (e.g., word count, formatting) or "loose" (e.g., tone, clarity, quality)\n' ...
        '- "description": what the constraint requires\n' ...
        '- "verifiable": true for strict, false for loose\n\n' ...
        'IMPORTANT: You MUST extract at least one "loose" constraint (verifiable=false) for overall quality/tone.\n\n' ...
        'Example: [{"type":"strict","description":"Use bullet points","verifiable":true}, {"type":"loose","description":"Engaging and humanlike tone","verifiable":false}]'], prompt);
    
    extracted = [];
    for attempt = 1:3
        try
            response = api.call(extraction_prompt, 0.1);
            extracted = safe_json_decode(response);
            
            % CRITICAL FIX: Ensure the decoded output is actually a struct array
            % with the required fields before accepting it.
            if isstruct(extracted) && ~isempty(extracted) && isfield(extracted, 'verifiable')
                break; 
            else
                extracted = []; % Reset to empty if it's a string/number
            end
        catch
            pause(1);
        end
    end
    
    % Fallback if all attempts fail to return valid JSON, or if the decoded
    % structs are missing any of the three required fields.
    if ~isstruct(extracted) || isempty(extracted) || ~isfield(extracted, 'verifiable') ...
            || ~isfield(extracted, 'type') || ~isfield(extracted, 'description')
        extracted = struct('type', 'loose', 'description', 'Overall clarity and relevance', 'verifiable', false);
    end

    % Canonicalize: force a 1xN row containing exactly the three expected
    % fields. jsondecode can return Nx1 columns or structs carrying extra
    % fields, either of which crashes concatenation with the injected
    % default loose constraint below (horzcat dimension/field mismatch).
    extracted = reshape(extracted, 1, []);
    extracted = struct('type', {extracted.type}, ...
                       'description', {extracted.description}, ...
                       'verifiable', {extracted.verifiable});
    
    % CRITICAL SAFETY CHECK: Ensure there is AT LEAST ONE loose constraint
    % If the LLM only returned strict constraints, we inject a default loose one.
    has_loose = any([extracted.verifiable] == 0);
    if ~has_loose
        new_loose = struct('type', 'loose', 'description', 'Overall clarity, coherence and tone', 'verifiable', false);
        extracted = [extracted, new_loose];
    end
end