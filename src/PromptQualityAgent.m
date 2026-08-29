function quality = PromptQualityAgent(prompt, api)
% PromptQualityAgent scores the prompt itself, independent of any response
% or of ExtractorAgent's output, to keep prompt-quality and response-compliance
% as separable constructs (see Prompt Response Divergence metric).

    quality = struct();
    quality.prompt = prompt;

    % --- Deterministic: Specificity and Completeness ---
    % Checklist derived from IFEval's constraint categories: length, format,
    % keyword, case, punctuation, start/end, combination.
    checklist = struct( ...
        'length',    {{'word', 'sentence', 'paragraph', 'character', 'words', 'sentences', 'paragraphs'}}, ...
        'format',    {{'bullet', 'list', 'table', 'json', 'markdown', 'heading', 'numbered'}}, ...
        'keyword',   {{'include', 'must contain', 'mention', 'avoid', 'do not use', 'without using'}}, ...
        'case',      {{'uppercase', 'lowercase', 'capitalize', 'all caps'}}, ...
        'structure', {{'introduction', 'conclusion', 'start with', 'end with', 'begin with'}}, ...
        'audience',  {{'for a', 'audience', 'beginner', 'expert', 'child', 'professional'}} ...
    );

    fields = fieldnames(checklist);
    p_lower = lower(prompt);
    hits = 0;
    for i = 1:numel(fields)
        terms = checklist.(fields{i});
        for j = 1:numel(terms)
            if contains(p_lower, terms{j})
                hits = hits + 1;
                break; % count category once, not per matched term
            end
        end
    end

    num_categories = numel(fields);
    quality.completeness = hits / num_categories; % fraction of constraint categories addressed

    % Specificity: presence of concrete, checkable detail vs vague language
    vague_terms = {'something', 'stuff', 'good', 'nice', 'anything', 'whatever', 'some'};
    vague_hits = 0;
    for i = 1:numel(vague_terms)
        if contains(p_lower, vague_terms{i})
            vague_hits = vague_hits + 1;
        end
    end
    word_count = numel(strsplit(prompt));
    % Longer, less vague prompts score higher specificity, capped at 1
    specificity_raw = (hits / max(1, num_categories)) - (vague_hits * 0.15);
    quality.specificity = max(0, min(1, specificity_raw + min(0.3, word_count / 200)));

    % --- LLM-judged: Ambiguity and Structure, ensembled ---
    % Escaping is now handled centrally inside Ollama_API.m
    judge_prompt = sprintf([ ...
        'Rate the following student-written prompt (not the response to it) on two dimensions,\n' ...
        'each on a scale of 1 to 5. Judge the prompt ALONE, before any response exists.\n\n' ...
        'Prompt: "%s"\n\n' ...
        'Dimensions:\n' ...
        '1. ambiguity: 1 = highly ambiguous / open to many interpretations, 5 = fully unambiguous\n' ...
        '2. structure: 1 = disorganized / unclear ordering of intent, 5 = clearly and logically structured\n\n' ...
        'Return ONLY JSON: {"ambiguity": X, "structure": Y}'], prompt);

    num_samples = 3;
    ambiguity_scores = [];
    structure_scores = [];
    for s = 1:num_samples
        resp = api.call(judge_prompt, 0.7); % temperature varied for ensembling diversity
        parsed = safe_json_decode(resp);
        if ~isempty(parsed) && isfield(parsed, 'ambiguity') && isfield(parsed, 'structure')
            ambiguity_scores(end+1) = parsed.ambiguity; %#ok<AGROW>
            structure_scores(end+1) = parsed.structure; %#ok<AGROW>
        end
    end

    if isempty(ambiguity_scores)
        quality.ambiguity = NaN;
        quality.structure = NaN;
        quality.llm_confidence = 0;
    else
        quality.ambiguity = mean(ambiguity_scores);
        quality.structure = mean(structure_scores);
        
        % Confidence derived from agreement across ensembled samples
        % Formula: C = 1 - (sigma^2 / V_max), where V_max = 4 for a 1-5 scale
        if numel(ambiguity_scores) > 1
            avg_variance = (var(ambiguity_scores) + var(structure_scores)) / 2;
            V_max = 4; % Max possible variance for a 1-5 scale
            quality.llm_confidence = max(0, 1 - (avg_variance / V_max));
        else
            quality.llm_confidence = 0.5; % only one sample succeeded, low confidence
        end
    end
end