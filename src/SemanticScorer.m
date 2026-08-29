function scored = SemanticScorer(prompt, response, extracted_constraints, api)
    scored = extracted_constraints;
    if isempty(scored), return; end
    num_c = length(scored);
    strict_idx = [scored.verifiable] == 1;
    loose_idx = [scored.verifiable] == 0;

    % --- Strict constraints: deterministic checks only, no silent pass ---
    for i = 1:num_c
        if strict_idx(i)
            desc = lower(scored(i).description);
            [result, checked] = check_strict_constraint(desc, response);
            if checked
                scored(i).score = result;
                scored(i).confidence = 1.0;
                scored(i).needs_review = false;
            else
                % Unrecognized strict constraint type: do NOT default to pass.
                % Route to manual review instead.
                scored(i).score = NaN;
                scored(i).confidence = 0;
                scored(i).needs_review = true;
            end
        end
    end

    % --- Loose constraints: ensembled LLM scoring ---
    if any(loose_idx)
        loose_c = scored(loose_idx);
        num_samples = 3;
        results_by_constraint = cell(1, length(loose_c));

        for s = 1:num_samples
            % Escaping is now handled centrally inside Ollama_API.m
            scoring_prompt = sprintf(['Rate how well this response satisfies EACH constraint on a scale of 1-5.\n\n' ...
                'Response: "%s"\n\nConstraints:\n'], response);
            for i = 1:length(loose_c)
                scoring_prompt = [scoring_prompt, sprintf('%d. %s\n', i, loose_c(i).description)]; %#ok<AGROW>
            end
            scoring_prompt = [scoring_prompt, '\nReturn ONLY a JSON array of objects: [{"score": X}, ...]'];

            llm_resp = api.call(scoring_prompt, 0.7); % varied temperature for ensembling
            result = safe_json_decode(llm_resp);
            if ~isempty(result)
                for i = 1:min(length(result), length(loose_c))
                    if isfield(result(i), 'score')
                        results_by_constraint{i}(end+1) = result(i).score; %#ok<AGROW>
                    end
                end
            end
        end

        l_idx = 1;
        V_max = 4; % Max possible variance for a 1-5 scale
        for i = 1:num_c
            if loose_idx(i)
                samples = results_by_constraint{l_idx};
                if ~isempty(samples)
                    scored(i).score = mean(samples);
                    if numel(samples) > 1
                        % Confidence formula: C = 1 - (sigma^2 / V_max)
                        variance = var(samples);
                        scored(i).confidence = max(0, 1 - (variance / V_max));
                    else
                        scored(i).confidence = 0.5; % only one sample succeeded
                    end
                else
                    scored(i).score = NaN;
                    scored(i).confidence = 0;
                end
                % Confidence-based routing: below 0.6, flag for human review
                % instead of forcing an automated score through the pipeline.
                scored(i).needs_review = scored(i).confidence < 0.6;
                l_idx = l_idx + 1;
            end
        end
    end
end

function [result, checked] = check_strict_constraint(desc, response)
% Deterministic verification for recognized strict constraint types.
% Returns checked=false for anything not in this list, so the caller
% routes it to manual review rather than assuming it passed.
    checked = true;
    if contains(desc, 'bullet') || contains(desc, 'list')
        result = double(contains(response, '*') || contains(response, '-') || ~isempty(regexp(response, '^\d+\.', 'once')));
    elseif contains(desc, 'word count') || (contains(desc, 'word') && ~isempty(regexp(desc, '\d+', 'once')))
        target = str2double(regexp(desc, '\d+', 'match', 'once'));
        if ~isnan(target)
            result = double(abs(length(strsplit(response)) - target) <= 5);
        else
            checked = false; result = NaN;
        end
    elseif contains(desc, 'character')
        target = str2double(regexp(desc, '\d+', 'match', 'once'));
        if ~isnan(target)
            result = double(abs(length(response) - target) <= 10);
        else
            checked = false; result = NaN;
        end
    elseif contains(desc, 'uppercase') || contains(desc, 'all caps')
        result = double(strcmp(response, upper(response)));
    elseif contains(desc, 'lowercase')
        result = double(strcmp(response, lower(response)));
    elseif contains(desc, 'must include') || contains(desc, 'must contain') || contains(desc, 'mention')
        keyword = regexp(desc, '"([^"]+)"', 'tokens', 'once');
        if ~isempty(keyword)
            result = double(contains(lower(response), lower(keyword{1})));
        else
            checked = false; result = NaN;
        end
    elseif contains(desc, 'must start with') || contains(desc, 'begin with')
        keyword = regexp(desc, '"([^"]+)"', 'tokens', 'once');
        if ~isempty(keyword)
            result = double(startsWith(strtrim(response), keyword{1}));
        else
            checked = false; result = NaN;
        end
    elseif contains(desc, 'must end with')
        keyword = regexp(desc, '"([^"]+)"', 'tokens', 'once');
        if ~isempty(keyword)
            result = double(endsWith(strtrim(response), keyword{1}));
        else
            checked = false; result = NaN;
        end
    else
        checked = false;
        result = NaN;
    end
end