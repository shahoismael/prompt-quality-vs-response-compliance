function run_ablation_factorial(in_name, base_name, out_name, fresh)
% Factorial arms for the ablation, on the SAME items run_baseline already used.
%
% The published ablation compares the assembled framework against a naive
% single-pass judge and a chain-of-thought judge. Those systems differ in four
% respects at once (rubric decomposition, three-sample ensembling, length
% calibration, confidence routing) and in budget (three model calls against
% one), so the comparison cannot attribute the difference to any one factor.
%
% This script adds the two arms that separate them:
%   ARM A  naive judge, 3 samples averaged   - ensembling at matched budget,
%                                              no rubric, no calibration
%   ARM B  rubric judge, 1 sample            - rubric at single-pass budget,
%                                              no ensembling
%
% With the existing naive (1 sample) and framework (3 samples, rubric) arms,
% the 2x2 is complete.
%
% RESUME
%   Every finished item is written to a checkpoint file next to the output,
%   together with the random-number state. If the computer shuts down, sleeps
%   or MATLAB is closed, run the SAME command again: finished items are
%   skipped and the run continues from the next one, drawing the same
%   perturbations an uninterrupted run would have drawn. At most the item in
%   progress is lost.
%
%   The checkpoint is written to a temporary file and then renamed, so a
%   power cut during the write cannot corrupt it.
%
%   If Ollama is not running, the script stops with a message instead of
%   recording empty scores. Start Ollama and run the same command again.
%
%   To discard the checkpoint and start over, pass fresh = true.
%
% RUN FROM THE PROJECT ROOT.
%   run_ablation_factorial('results/simulation_results_main500_v2.mat', ...
%                          'results/baseline_results_v2.mat', ...
%                          'results/ablation_factorial.mat')

    if nargin < 1 || isempty(in_name),   in_name   = 'results/simulation_results_main500_v2.mat'; end
    if nargin < 2 || isempty(base_name), base_name = 'results/baseline_results_v2.mat'; end
    if nargin < 3 || isempty(out_name),  out_name  = 'results/ablation_factorial.mat'; end
    if nargin < 4 || isempty(fresh),     fresh     = false; end

    assert(exist(in_name,   'file') == 2, ['Not found: ' in_name '. Run from the PROJECT ROOT.']);
    assert(exist(base_name, 'file') == 2, ['Not found: ' base_name '. Run run_baseline first.']);

    [out_dir, out_stem] = fileparts(out_name);
    if isempty(out_dir), out_dir = '.'; end
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    ckpt     = fullfile(out_dir, [out_stem '_checkpoint.mat']);
    ckpt_tmp = fullfile(out_dir, [out_stem '_checkpoint_tmp.mat']);

    % A finished run is never overwritten by accident.
    if exist(out_name, 'file') == 2 && ~fresh
        fprintf('Already complete: %s exists. Pass fresh = true to run again.\n', out_name);
        return;
    end
    if fresh
        if exist(ckpt, 'file') == 2, delete(ckpt); end
        if exist(ckpt_tmp, 'file') == 2, delete(ckpt_tmp); end
    end

    api = Ollama_API('qwen2.5:7b');
    require_ollama(api);

    S = load(in_name);   raw = S.raw_results;
    B = load(base_name);
    sel = B.subset_index;              % the identical 120-item subsample
    raw = raw(sel);
    n = numel(raw);

    rec = struct('src_index', {}, 'source', {}, ...
        'naive3_score', {}, 'naive3_bv', {}, 'naive3_lat_one', {}, ...
        'rub1_score',   {}, 'rub1_bv',   {}, 'rub1_lat_one',   {});

    if exist(ckpt, 'file') == 2
        C = load(ckpt);
        assert(isequal(C.sel(:), sel(:)), ...
            'Checkpoint was made on a different item subset. Run with fresh = true to start over.');
        rec = C.rec;
        rng(C.rng_state);
        fprintf('Resuming: %d of %d items already done.\n', numel(rec), n);
    else
        rng(42, 'twister');
        fprintf('Factorial ablation on %d items (same subsample as run_baseline)...\n', n);
    end

    t_session = tic; done_session = 0;
    for i = numel(rec) + 1 : n
        t_item = tic;
        p  = raw(i).prompt;
        r  = raw(i).baseline_resp;
        pA = perturb_prompt(p, 'A');
        pB = perturb_prompt(p, 'B');
        rC = perturb_response(r);

        % ---- ARM A: naive judge, 3-sample ensemble (matched budget) ----
        t0 = tic;  a_o = naive3(p,  r,  api);  lat_a = toc(t0);
        a_a = naive3(pA, r,  api);
        a_b = naive3(pB, r,  api);
        a_c = naive3(p,  rC, api);
        v = [a_o, a_a, a_b, a_c]; v = v(~isnan(v));
        if numel(v) > 1, bv_a = var(v); else, bv_a = NaN; end

        % ---- ARM B: rubric judge, single sample ----
        t0 = tic;  b_o = rubric1(p,  r,  api);  lat_b = toc(t0);
        b_a = rubric1(pA, r,  api);
        b_b = rubric1(pB, r,  api);
        b_c = rubric1(p,  rC, api);
        v = [b_o, b_a, b_b, b_c]; v = v(~isnan(v));
        if numel(v) > 1, bv_b = var(v); else, bv_b = NaN; end

        % Every score empty: check the server before recording anything, so a
        % stopped Ollama cannot fill the checkpoint with empty items.
        if all(isnan([a_o a_a a_b a_c b_o b_a b_b b_c]))
            require_ollama(api, sprintf('Progress is saved through item %d of %d.', i - 1, n));
        end

        rec(end+1) = struct('src_index', sel(i), 'source', raw(i).source, ...
            'naive3_score', a_o, 'naive3_bv', bv_a, 'naive3_lat_one', lat_a, ...
            'rub1_score',   b_o, 'rub1_bv',   bv_b, 'rub1_lat_one',   lat_b); %#ok<AGROW>

        % Checkpoint after EVERY item: write to a temp file, then rename.
        rng_state = rng; %#ok<NASGU>
        save(ckpt_tmp, 'rec', 'sel', 'rng_state', '-v7');
        movefile(ckpt_tmp, ckpt, 'f');

        done_session = done_session + 1;
        per_item = toc(t_session) / done_session;
        fprintf('Factorial: %d/%d  (%.0f s, about %.1f h left)\n', ...
            i, n, toc(t_item), per_item * (n - i) / 3600);
    end

    factorial_results = struct2table(rec); %#ok<NASGU>
    save(out_name, 'factorial_results', 'sel', '-v7.3');
    if exist(ckpt, 'file') == 2, delete(ckpt); end

    s_a = [rec.naive3_score]; s_b = [rec.rub1_score];
    fprintf('\n===== FACTORIAL ARMS =====\n');
    fprintf('ARM A naive x3 : M=%.2f SD=%.2f  ceiling share=%.3f  BV Mdn=%.3f\n', ...
        mean(s_a,'omitnan'), std(s_a,'omitnan'), mean(s_a==5), median([rec.naive3_bv],'omitnan'));
    fprintf('ARM B rubric x1: M=%.2f SD=%.2f  ceiling share=%.3f  BV Mdn=%.3f\n', ...
        mean(s_b,'omitnan'), std(s_b,'omitnan'), mean(s_b==5), median([rec.rub1_bv],'omitnan'));
    fprintf('Compare against naive x1 and the framework in %s\n', base_name);
    fprintf('===== END =====\n');
end

function require_ollama(api, context)
% Stop with a clear message if the local server does not answer.
    if nargin < 2, context = ''; end
    probe = '';
    for t = 1:3
        probe = api.call('Reply with the single word OK.', 0.0);
        if ~isempty(probe), return; end
        pause(10);
    end
    error('Ollama is not responding at %s. Start Ollama, then run the same command again. %s', ...
        api.endpoint, context);
end

function out = call_retry(api, q, temp)
% Retry a call that returned nothing (dropped connection, e.g. after sleep).
% A reply that arrives but does not parse is NOT retried, matching run_baseline.
    out = '';
    for t = 1:3
        out = api.call(q, temp);
        if ~isempty(out), return; end
        pause(10);
    end
end

function score = naive3(prompt, response, api)
% Naive judge prompt, identical wording to run_baseline, averaged over 3 draws.
    v = nan(1,3);
    for k = 1:3
        q = sprintf(['Rate how well the following response satisfies the request, ' ...
            'on a scale of 1 to 5.\n\nRequest: "%s"\n\nResponse: "%s"\n\n' ...
            'Return ONLY JSON: {"score": X}'], prompt, response);
        v(k) = extract_score(call_retry(api, q, 0.7));
    end
    v = v(~isnan(v));
    if isempty(v), score = NaN; else, score = mean(v); end
end

function score = rubric1(prompt, response, api)
% Rubric decomposition at single-pass budget: the response is rated against
% the criteria the Extractor derives from the prompt, one draw, no calibration.
    try
        crit = ExtractorAgent(prompt, api);
    catch
        score = NaN; return;
    end
    if isempty(crit) || ~isfield(crit, 'loose') || isempty(crit.loose)
        items = {'clarity, coherence and tone'};
    else
        items = crit.loose;
    end
    lines = '';
    for k = 1:numel(items)
        c = items{k};
        if ~ischar(c) && ~isstring(c), c = char(string(c)); end
        lines = [lines sprintf('- %s\n', c)]; %#ok<AGROW>
    end
    q = sprintf(['Rate how well the response satisfies EACH criterion below, ' ...
        'then give one overall score from 1 to 5.\n\nCriteria:\n%s\n' ...
        'Response: "%s"\n\nReturn ONLY JSON: {"score": X}'], lines, response);
    score = extract_score(call_retry(api, q, 0.7));
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
