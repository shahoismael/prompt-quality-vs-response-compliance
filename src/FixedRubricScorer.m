function fixed = FixedRubricScorer(response, api)
% Prompt-INDEPENDENT scoring of a generated response.
%
% Why this exists. In the main pipeline, response compliance is scored
% against criteria the Extractor derived FROM the prompt. That makes the
% scoring criteria endogenous to prompt quality: an underspecified prompt
% yields few, weak, easily satisfied criteria (high compliance), while a
% well-specified prompt yields many strict ones (lower compliance). Any
% correlation between prompt quality and compliance is therefore suppressed
% by construction, and a near-zero result cannot be distinguished from an
% artefact of the design.
%
% This agent breaks that coupling. It never sees the prompt. It rates the
% response as a standalone text on three prompt-independent dimensions, so
% the prompt-quality/response-quality correlation can be estimated without
% a moving denominator. If the correlation is still near zero here, the
% divergence finding survives its strongest objection.
%
% Returns: struct with coherence, informativeness, clarity, composite,
% confidence, and n_samples.

    fixed = struct('coherence', NaN, 'informativeness', NaN, 'clarity', NaN, ...
        'composite', NaN, 'confidence', 0, 'n_samples', 0);

    if isempty(response), return; end

    judge_prompt = sprintf([ ...
        'Rate the following TEXT on three dimensions, each 1 to 5.\n' ...
        'Judge the text entirely on its own merits. You have not been shown\n' ...
        'the request that produced it, and you must not speculate about it.\n\n' ...
        'Text: "%s"\n\n' ...
        'Dimensions:\n' ...
        '1. coherence: 1 = disorganised, self-contradictory, 5 = well organised and internally consistent\n' ...
        '2. informativeness: 1 = vacuous or padded, 5 = substantive and specific\n' ...
        '3. clarity: 1 = confusing or badly written, 5 = precise and easy to follow\n\n' ...
        'Return ONLY JSON: {"coherence": X, "informativeness": Y, "clarity": Z}'], response);

    num_samples = 3;
    coh = []; inf_ = []; cla = [];
    for s = 1:num_samples
        raw = api.call(judge_prompt, 0.7);
        parsed = safe_json_decode(raw);
        if ~isempty(parsed) && isfield(parsed, 'coherence') && ...
                isfield(parsed, 'informativeness') && isfield(parsed, 'clarity')
            coh(end+1)  = parsed.coherence;      %#ok<AGROW>
            inf_(end+1) = parsed.informativeness; %#ok<AGROW>
            cla(end+1)  = parsed.clarity;        %#ok<AGROW>
        end
    end

    if isempty(coh), return; end

    fixed.coherence       = mean(coh);
    fixed.informativeness = mean(inf_);
    fixed.clarity         = mean(cla);
    fixed.composite       = mean([fixed.coherence, fixed.informativeness, fixed.clarity]);
    fixed.n_samples       = numel(coh);

    if numel(coh) > 1
        avg_var = (var(coh) + var(inf_) + var(cla)) / 3;
        fixed.confidence = max(0, 1 - (avg_var / 4)); % V_max = 4 on a 1-5 scale
    else
        fixed.confidence = 0.5;
    end
end
