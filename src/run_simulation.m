function run_simulation(num_test_prompts, out_name)
% MAIN PIPELINE.
%
% RUN FROM THE PROJECT ROOT, NOT FROM src/:
%       cd D:\claude_projects\R3
%       addpath('src')
%       run_simulation(500, 'results/simulation_results_main500_v2.mat')
%
% Every path below ('data/...', 'results/...') is relative to the working
% directory. Launching from src/ resolves them to src/data and src/results,
% which do not exist. That is why earlier runs produced no output.
%
% Changes in v2, all of which move reported numbers:
%   1. Perturbation variants are scored against the ORIGINAL extracted
%      constraints. Previously the Extractor was re-run on each perturbed
%      prompt, so "bias variance" was the spread of scores across DIFFERENT
%      criteria, not the stability of one judge on one criterion set. Bias
%      variance is now paired per constraint, and two calls per item are saved.
%   2. Confidence routing is APPLIED, not merely counted. Judgments with
%      needs_review = true are excluded from the reported score. Previously
%      every low-confidence judgment still entered every published statistic.
%   3. Raw and calibrated compliance are both retained, so the calibration
%      step can be shown not to have created the divergence result.
%   4. A prompt-INDEPENDENT rubric arm (FixedRubricScorer) is added. Criteria
%      extracted from the prompt are endogenous to prompt quality; this arm
%      gives a compliance measure whose criteria do not move with the prompt.
%   5. Strict accuracy separates FAILED from UNVERIFIABLE. Previously an
%      unrecognised strict constraint was stored as NaN and mean(NaN==1)
%      scored it 0, i.e. counted it as a failure. The definitive figure now
%      comes from verify_ifeval.py using the official IFEval registry.
%   6. Spearman is tie-corrected (spearman_ties.m).
%   7. raw_results is ALWAYS saved, so the run can be re-analysed without
%      regenerating a single response. Partial checkpoints every 25 items.

    if nargin < 1 || isempty(num_test_prompts), num_test_prompts = 5; end
    if nargin < 2 || isempty(out_name), out_name = 'results/simulation_results.mat'; end

    assert(exist('data/master_benchmark.json', 'file') == 2, ...
        ['data/master_benchmark.json not found. Run from the PROJECT ROOT ' ...
         '(cd D:\\claude_projects\\R3; addpath(''src'')), not from src/.']);
    if ~isfolder('results'), mkdir('results'); end

    rng(42, 'twister');
    api = Ollama_API('qwen2.5:7b');

    data    = jsondecode(fileread('data/master_benchmark.json'));
    prompts = data.prompts;
    sources = data.source;

    idx     = randperm(numel(prompts));
    prompts = prompts(idx);
    sources = sources(idx);

    n = min(num_test_prompts, numel(prompts));
    prompts = prompts(1:n);
    sources = sources(1:n);

    fprintf('Running %d prompts. Output: %s\n', n, out_name);
    run_started = datetime('now');

    raw_results = struct('prompt', {}, 'source', {}, 'baseline_resp', {}, ...
        'extracted_constraints', {}, 'scored_orig', {}, 'scored_A', {}, ...
        'scored_B', {}, 'scored_C', {}, 'prompt_quality', {}, 'fixed_rubric', {}, ...
        'strict_acc_checked', {}, 'strict_n_total', {}, 'strict_n_unverifiable', {}, ...
        'loose_acc', {}, 'bias_variance', {}, 'raw_loose_kept', {}, ...
        'n_loose', {}, 'n_routed', {}, 'latency', {});

    failures = 0;

    % ================= PASS 1: generate and score =================
    for i = 1:n
        item_timer  = tic;
        orig_prompt = prompts{i};
        source      = sources{i};

        try
            % Agents 1 and 2 are independent: the Prompt Quality Agent never
            % sees the Extractor's output, so prompt quality cannot be a
            % restatement of extraction quality.
            extracted = normalize_constraints(ExtractorAgent(orig_prompt, api));
            pq        = PromptQualityAgent(orig_prompt, api);

            if isempty(extracted)
                fprintf('Row %d: Extractor returned empty. Skipping.\n', i);
                failures = failures + 1; continue;
            end

            baseline_resp = api.call(['Respond to: ' orig_prompt], 0.2);
            if isempty(baseline_resp)
                fprintf('Row %d: generation failed. Skipping.\n', i);
                failures = failures + 1; continue;
            end

            scored_orig = SemanticScorer(orig_prompt, baseline_resp, extracted, api);

            % ---- perturbations, all against the ORIGINAL constraint set ----
            pA       = perturb_prompt(orig_prompt, 'A');   % typos
            scored_A = SemanticScorer(pA, baseline_resp, extracted, api);

            pB       = perturb_prompt(orig_prompt, 'B');   % formatting only
            scored_B = SemanticScorer(pB, baseline_resp, extracted, api);

            respC    = perturb_response(baseline_resp);    % verbosity
            scored_C = SemanticScorer(orig_prompt, respC, extracted, api);

            % ---- prompt-independent compliance arm ----
            fixed_rubric = FixedRubricScorer(baseline_resp, api);

            % ---- bias variance: paired, per constraint, across 4 conditions ----
            L  = getflag(scored_orig, 'verifiable') == 0;
            s0 = getnum(scored_orig, 'score');
            sA = getnum(scored_A, 'score');
            sB = getnum(scored_B, 'score');
            sC = getnum(scored_C, 'score');
            bias_variance = NaN;
            if any(L)
                M = [s0(L); sA(L); sB(L); sC(L)];
                per_c = nan(1, size(M,2));
                for c = 1:size(M,2)
                    col = M(:,c); col = col(~isnan(col));
                    if numel(col) > 1, per_c(c) = var(col); end
                end
                bias_variance = mean(per_c, 'omitnan');
            end

            % ---- confidence routing, APPLIED ----
            review   = getflag(scored_orig, 'needs_review') == 1;
            keep     = L & ~review;
            n_loose  = sum(L);
            n_routed = sum(L & review);

            kept = s0(keep); kept = kept(~isnan(kept));
            if isempty(kept), raw_loose_kept = NaN; else, raw_loose_kept = mean(kept); end

            % ---- strict verification (provisional; see verify_ifeval.py) ----
            strict_acc_checked = NaN; strict_n_total = 0; strict_n_unverifiable = 0;
            loose_acc = NaN;
            if strcmp(source, 'ifeval')
                Sx = s0(getflag(scored_orig, 'verifiable') == 1);
                strict_n_total        = numel(Sx);
                strict_n_unverifiable = sum(isnan(Sx));
                checked = Sx(~isnan(Sx));
                if ~isempty(checked), strict_acc_checked = mean(checked == 1); end
                if ~isempty(kept),    loose_acc = mean(kept >= 4); end
            end

            raw_results(end+1) = struct( ... %#ok<AGROW>
                'prompt', orig_prompt, 'source', source, ...
                'baseline_resp', baseline_resp, 'extracted_constraints', extracted, ...
                'scored_orig', scored_orig, 'scored_A', scored_A, ...
                'scored_B', scored_B, 'scored_C', scored_C, ...
                'prompt_quality', pq, 'fixed_rubric', fixed_rubric, ...
                'strict_acc_checked', strict_acc_checked, ...
                'strict_n_total', strict_n_total, ...
                'strict_n_unverifiable', strict_n_unverifiable, ...
                'loose_acc', loose_acc, 'bias_variance', bias_variance, ...
                'raw_loose_kept', raw_loose_kept, 'n_loose', n_loose, ...
                'n_routed', n_routed, 'latency', toc(item_timer));

            fprintf('Pass 1: %d/%d  (%.0f s, routed %d/%d)\n', ...
                i, n, raw_results(end).latency, n_routed, n_loose);

            % checkpoint: a 50-hour run must survive a crash
            if mod(numel(raw_results), 25) == 0
                save([out_name '.partial'], 'raw_results', '-v7.3');
                fprintf('  [checkpoint: %d items saved]\n', numel(raw_results));
            end

        catch item_err
            failures = failures + 1;
            fprintf('Row %d: error (%s). Skipping.\n', i, item_err.message);
        end
    end

    if isempty(raw_results)
        error('No items completed. Check that Ollama is running: diagnose_ollama');
    end

    % ================= PASS 2: calibrate across the batch =================
    all_responses  = {raw_results.baseline_resp};
    all_raw_scores = [raw_results.raw_loose_kept];

    vn = {'Prompt','Source','Strict_Acc_Checked','Strict_N_Unverifiable', ...
          'Loose_Acc','Bias_Variance','Raw_Loose_Score','Calibrated_Loose_Score', ...
          'FixedRubric_Score','Prompt_Specificity','Prompt_Completeness', ...
          'Prompt_Ambiguity_Structure','Prompt_Composite4','N_Loose','N_Routed','Latency_s'};
    vt = [{'cell','cell'}, repmat({'double'}, 1, 14)];
    results = table('Size', [0, numel(vn)], 'VariableTypes', vt, 'VariableNames', vn);

    cal_info = struct('fitted', false, 'beta0', NaN, 'beta1', NaN, ...
        'se_beta1', NaN, 'r2', NaN, 'n', 0, 'mean_length', NaN);

    for i = 1:numel(raw_results)
        r = raw_results(i);
        [calibrated, ci_i] = CalibrationAgent(r.scored_orig, r.baseline_resp, ...
            all_responses, all_raw_scores);
        if ci_i.fitted, cal_info = ci_i; end

        L      = getflag(calibrated, 'verifiable') == 0;
        review = getflag(r.scored_orig, 'needs_review') == 1;
        keep   = L & ~review;
        cal_kept = getnum(calibrated, 'score');
        cal_kept = cal_kept(keep);
        cal_kept = cal_kept(~isnan(cal_kept));
        if isempty(cal_kept), cal_mean = NaN; else, cal_mean = mean(cal_kept); end

        pq         = r.prompt_quality;
        pq_amb_str = mean([pq.ambiguity, pq.structure], 'omitnan');
        pq_comp4   = mean([pq.specificity, pq.completeness, pq.ambiguity, pq.structure], 'omitnan');

        results = [results; table({r.prompt}, {r.source}, ...
            r.strict_acc_checked, r.strict_n_unverifiable, r.loose_acc, ...
            r.bias_variance, r.raw_loose_kept, cal_mean, r.fixed_rubric.composite, ...
            pq.specificity, pq.completeness, pq_amb_str, pq_comp4, ...
            r.n_loose, r.n_routed, r.latency, 'VariableNames', vn)]; %#ok<AGROW>
    end

    % ================= divergence, tie-corrected, every variant =================
    div = struct();
    pairs = { ...
        'amb_vs_cal',     results.Prompt_Ambiguity_Structure, results.Calibrated_Loose_Score; ...
        'amb_vs_raw',     results.Prompt_Ambiguity_Structure, results.Raw_Loose_Score; ...
        'amb_vs_fixed',   results.Prompt_Ambiguity_Structure, results.FixedRubric_Score; ...
        'comp4_vs_cal',   results.Prompt_Composite4,          results.Calibrated_Loose_Score; ...
        'comp4_vs_fixed', results.Prompt_Composite4,          results.FixedRubric_Score};
    for k = 1:size(pairs,1)
        [rho, p, ci, nn] = spearman_ties(pairs{k,2}, pairs{k,3});
        div.(pairs{k,1}) = struct('rho', rho, 'p', p, 'ci', ci, 'n', nn);
    end

    % ================= export IFEval items for the official verifier =========
    fid = fopen('results/ifeval_responses.jsonl', 'w', 'n', 'UTF-8');
    n_exported = 0;
    for i = 1:numel(raw_results)
        if strcmp(raw_results(i).source, 'ifeval')
            fprintf(fid, '%s\n', jsonencode(struct( ...
                'prompt',   raw_results(i).prompt, ...
                'response', raw_results(i).baseline_resp)));
            n_exported = n_exported + 1;
        end
    end
    fclose(fid);

    run_finished = datetime('now');
    save(out_name, 'results', 'raw_results', 'div', 'cal_info', ...
        'run_started', 'run_finished', '-v7.3');
    if exist([out_name '.partial'], 'file'), delete([out_name '.partial']); end

    % ================= summary =================
    fprintf('\n===== RUN COMPLETE =====\n');
    fprintf('Evaluated %d/%d (%d failures). Elapsed: %s\n', ...
        height(results), n, failures, string(run_finished - run_started));
    fprintf('Routing: %d/%d loose judgments deferred (%.1f%%)\n', ...
        sum(results.N_Routed), sum(results.N_Loose), ...
        100*sum(results.N_Routed)/max(1,sum(results.N_Loose)));
    fprintf('Items with NO automated score after routing: %d\n', ...
        sum(isnan(results.Calibrated_Loose_Score)));
    fprintf('Strict (provisional, checked only): %.4f | unverifiable dropped: %d\n', ...
        mean(results.Strict_Acc_Checked,'omitnan'), sum(results.Strict_N_Unverifiable));
    fprintf('Loose acc: %.4f | BV mean %.4f median %.4f\n', ...
        mean(results.Loose_Acc,'omitnan'), ...
        mean(results.Bias_Variance,'omitnan'), median(results.Bias_Variance,'omitnan'));
    if cal_info.fitted
        fprintf('Calibration: beta1 = %.6f (SE %.6f), t = %.2f, R2 = %.4f, n = %d\n', ...
            cal_info.beta1, cal_info.se_beta1, ...
            cal_info.beta1/max(eps,cal_info.se_beta1), cal_info.r2, cal_info.n);
    else
        fprintf('Calibration: NOT FITTED\n');
    end
    fprintf('\nDivergence (tie-corrected Spearman):\n');
    f = fieldnames(div);
    for k = 1:numel(f)
        d = div.(f{k});
        fprintf('  %-16s rho = %+.4f  95%% CI [%+.3f, %+.3f]  p = %.4f  n = %d\n', ...
            f{k}, d.rho, d.ci(1), d.ci(2), d.p, d.n);
    end
    fprintf('\n  amb_vs_fixed is the decisive one: its criteria do not come from\n');
    fprintf('  the prompt. If it is flat, the divergence claim survives.\n');
    fprintf('\nExported %d IFEval responses to results/ifeval_responses.jsonl\n', n_exported);
    fprintf('Next: python verify_ifeval.py   (definitive strict/loose accuracy)\n');
    fprintf('========================\n');
end

% ----------------------------------------------------------------------
function c = normalize_constraints(c)
% Coerce 'verifiable' to logical. jsondecode yields logical for JSON true,
% but a model that emits "true" as a STRING would otherwise poison every
% [c.verifiable] concatenation downstream.
    if isempty(c), return; end
    for i = 1:numel(c)
        v = c(i).verifiable;
        if ischar(v) || isstring(v)
            c(i).verifiable = any(strcmpi(string(v), ["true","1","yes"]));
        elseif isempty(v)
            c(i).verifiable = false;
        else
            c(i).verifiable = logical(v);
        end
    end
end

function v = getnum(s, field)
% Numeric field of a struct array as a 1xN row, NaN where missing or empty.
% Direct [s.field] silently DROPS empty elements and returns a short vector,
% which would misalign every logical mask built from a different field.
    v = nan(1, numel(s));
    if isempty(s) || ~isfield(s, field), return; end
    for i = 1:numel(s)
        x = s(i).(field);
        if ~isempty(x) && isnumeric(x) && isscalar(x), v(i) = double(x); end
    end
end

function v = getflag(s, field)
% Logical/0-1 field as a 1xN double row, 0 where missing or empty.
    v = zeros(1, numel(s));
    if isempty(s) || ~isfield(s, field), return; end
    for i = 1:numel(s)
        x = s(i).(field);
        if ~isempty(x) && (islogical(x) || isnumeric(x)) && isscalar(x)
            v(i) = double(x ~= 0);
        end
    end
end
