function run_baseline(in_name, out_name, subset_n)
% Scores Baseline 1 (naive single-pass) and Baseline 2 (chain-of-thought)
% on the SAME prompts and responses the framework already evaluated.
%
% RUN FROM THE PROJECT ROOT.
%   run_baseline('results/simulation_results_main500_v2.mat', ...
%                'results/baseline_results_v2.mat', 120)
%
% FIX (v2) - latency was not comparable. The framework figure was a whole-item
% total (response generation + 3-sample prompt-quality + four 3-sample scoring
% passes + the fixed-rubric arm), while each baseline figure was ONE judge
% call. The reported "4.5x" therefore compared ~20 calls against 1. This
% version records, per item and per system:
%     *_lat_one   - a single original-condition judgment
%     *_lat_total - all four perturbation-condition judgments
% so the paper can compare judgment-for-judgment. Report MEDIANS: a single
% model-reload stall in the previous run (6,156 s on one call) moved the
% naive mean from 42.5 s to 94.4 s and made CoT appear faster than naive,
% which it is not.

    if nargin < 1 || isempty(in_name),  in_name  = 'results/simulation_results.mat'; end
    if nargin < 2 || isempty(out_name), out_name = 'results/baseline_results.mat'; end
    if nargin < 3, subset_n = []; end

    assert(exist(in_name, 'file') == 2, ...
        ['Not found: ' in_name '. Run from the PROJECT ROOT and run_simulation first.']);

    rng(42, 'twister');
    api = Ollama_API('qwen2.5:7b');

    S = load(in_name);
    assert(isfield(S, 'raw_results'), 'raw_results missing. Re-run run_simulation (v2 always saves it).');
    raw = S.raw_results;

    % Deterministic subset so the ablation reuses the main run instead of
    % paying for a second framework pass over fresh items.
    if ~isempty(subset_n) && subset_n < numel(raw)
        sel = sort(randperm(numel(raw), subset_n));
        raw = raw(sel);
    else
        sel = 1:numel(raw);
    end
    n = numel(raw);
    fprintf('Scoring %d items with 2 baseline judges...\n', n);

    base = struct('prompt', {}, 'source', {}, 'src_index', {}, ...
        'naive_score', {}, 'naive_bv', {}, 'naive_lat_one', {}, 'naive_lat_total', {}, ...
        'cot_score', {},   'cot_bv', {},   'cot_lat_one', {},   'cot_lat_total', {});

    for i = 1:n
        p  = raw(i).prompt;
        r  = raw(i).baseline_resp;
        pA = perturb_prompt(p, 'A');
        pB = perturb_prompt(p, 'B');
        rC = perturb_response(r);

        % ---- Baseline 1: naive single-pass ----
        t_all = tic;
        t0 = tic;  s_o = naive_judge(p,  r,  api);  lat_one_naive = toc(t0);
        s_a = naive_judge(pA, r,  api);
        s_b = naive_judge(pB, r,  api);
        s_c = naive_judge(p,  rC, api);
        lat_tot_naive = toc(t_all);
        v = [s_o, s_a, s_b, s_c]; v = v(~isnan(v));
        if numel(v) > 1, bv_naive = var(v); else, bv_naive = NaN; end

        % ---- Baseline 2: chain-of-thought ----
        t_all = tic;
        t0 = tic;  c_o = cot_judge(p,  r,  api);  lat_one_cot = toc(t0);
        c_a = cot_judge(pA, r,  api);
        c_b = cot_judge(pB, r,  api);
        c_c = cot_judge(p,  rC, api);
        lat_tot_cot = toc(t_all);
        v = [c_o, c_a, c_b, c_c]; v = v(~isnan(v));
        if numel(v) > 1, bv_cot = var(v); else, bv_cot = NaN; end

        base(end+1) = struct('prompt', p, 'source', raw(i).source, ...
            'src_index', sel(i), ...
            'naive_score', s_o, 'naive_bv', bv_naive, ...
            'naive_lat_one', lat_one_naive, 'naive_lat_total', lat_tot_naive, ...
            'cot_score', c_o, 'cot_bv', bv_cot, ...
            'cot_lat_one', lat_one_cot, 'cot_lat_total', lat_tot_cot); %#ok<AGROW>

        fprintf('Baselines: %d/%d\n', i, n);
        if mod(i, 25) == 0
            baseline_results = struct2table(base); %#ok<NASGU>
            save([out_name '.partial'], 'baseline_results', '-v7.3');
        end
    end

    baseline_results = struct2table(base);
    subset_index = sel; %#ok<NASGU>
    if ~exist('results', 'dir'), mkdir('results'); end
    save(out_name, 'baseline_results', 'subset_index', '-v7.3');
    if exist([out_name '.partial'], 'file'), delete([out_name '.partial']); end
    fprintf('Saved %s\n', out_name);
end

function score = naive_judge(prompt, response, api)
    q = sprintf(['Rate how well the following response satisfies the request, ' ...
        'on a scale of 1 to 5.\n\nRequest: "%s"\n\nResponse: "%s"\n\n' ...
        'Return ONLY JSON: {"score": X}'], prompt, response);
    score = extract_score(api.call(q, 0.7));
end

function score = cot_judge(prompt, response, api)
    q = sprintf(['Evaluate how well the following response satisfies the request.\n\n' ...
        'Request: "%s"\n\nResponse: "%s"\n\n' ...
        'First, think step by step: identify what the request asked for and whether ' ...
        'the response delivers each element. Then, on the FINAL line, output ONLY ' ...
        'JSON: {"score": X} where X is an integer from 1 to 5.'], prompt, response);
    score = extract_score(api.call(q, 0.7));
end

function score = extract_score(llm_resp)
    score = NaN;
    if isempty(llm_resp), return; end
    matches = regexp(llm_resp, '\{[^{}]*"score"[^{}]*\}', 'match');
    if isempty(matches), return; end
    parsed = safe_json_decode(matches{end});
    if ~isempty(parsed) && isfield(parsed, 'score') && isnumeric(parsed.score)
        s = double(parsed.score);
        if s >= 1 && s <= 5, score = s; end
    end
end
