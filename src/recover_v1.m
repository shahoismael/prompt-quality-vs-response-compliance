function recover_v1(in_name)
% RE-ANALYSES THE EXISTING v1 ABLATION RUN. No inference, no Ollama, seconds.
%
%   cd D:\claude_projects\R3
%   addpath('src')
%   recover_v1
%
% results/simulation_results.mat (the 118-item run) is the ONLY old result
% file that saved raw_results, so it is the only one that can be re-analysed
% without regenerating responses. simulation_results_main500.mat holds the
% summary table alone and cannot be recovered.
%
% This quantifies three defects using data already on disk:
%
%   1. THE ROUTER WAS NEVER APPLIED. needs_review was computed and counted,
%      then every low-confidence judgment still entered the reported score.
%      Prints rho with and without the filter.
%
%   2. CALIBRATION MAY HAVE CREATED THE NULL. Response length plausibly rises
%      with prompt quality, so regressing length out can remove the very
%      variance being tested. Prints rho against raw and calibrated scores.
%
%   3. UNVERIFIABLE CONSTRAINTS WERE SCORED AS FAILURES. strict accuracy used
%      mean(score == 1) over a vector containing NaN for every constraint the
%      keyword checker could not evaluate, and NaN == 1 is false. Prints the
%      figure both ways plus the number of NaNs, so the size of the artefact
%      is visible before the new run confirms it.

    if nargin < 1 || isempty(in_name), in_name = 'results/simulation_results.mat'; end
    assert(exist(in_name,'file') == 2, ...
        ['Not found: ' in_name '. Run from the PROJECT ROOT.']);

    S = load(in_name);
    assert(isfield(S,'raw_results'), 'This file has no raw_results; it cannot be recovered.');
    raw = S.raw_results;
    n = numel(raw);
    fprintf('\n===== v1 RECOVERY: %s (%d items) =====\n', in_name, n);
    if isfield(S,'divergence_rho')
        fprintf('rho as reported in the manuscript: %.4f\n', S.divergence_rho);
    end

    raw_all = nan(1,n); raw_kept = nan(1,n);
    cal_all = nan(1,n); cal_kept = nan(1,n);
    pq_amb  = nan(1,n); pq_c4   = nan(1,n);
    n_loose = zeros(1,n); n_rout = zeros(1,n);
    sa_nan_as_fail = nan(1,n); sa_checked = nan(1,n);
    n_unverifiable = zeros(1,n); n_strict = zeros(1,n);
    is_ife = false(1,n);

    all_resp = {raw.baseline_resp};

    % batch score vector for the calibration fit, using the UNFILTERED mean
    % so the fit matches what v1 actually did
    batch = nan(1,n);
    for i = 1:n
        sc = raw(i).scored_orig;
        L  = getflag(sc,'verifiable') == 0;
        s  = getnum(sc,'score');
        v  = s(L); v = v(~isnan(v));
        if ~isempty(v), batch(i) = mean(v); end
    end

    for i = 1:n
        sc  = raw(i).scored_orig;
        L   = getflag(sc,'verifiable') == 0;
        rev = getflag(sc,'needs_review') == 1;
        s   = getnum(sc,'score');

        n_loose(i) = sum(L);
        n_rout(i)  = sum(L & rev);

        a = s(L);        a = a(~isnan(a));   if ~isempty(a), raw_all(i)  = mean(a); end
        k = s(L & ~rev); k = k(~isnan(k));   if ~isempty(k), raw_kept(i) = mean(k); end

        cal = CalibrationAgent(sc, raw(i).baseline_resp, all_resp, batch);
        cs  = getnum(cal,'score');
        a = cs(L);        a = a(~isnan(a));  if ~isempty(a), cal_all(i)  = mean(a); end
        k = cs(L & ~rev); k = k(~isnan(k));  if ~isempty(k), cal_kept(i) = mean(k); end

        pq = raw(i).prompt_quality;
        pq_amb(i) = mean([pq.ambiguity, pq.structure],'omitnan');
        pq_c4(i)  = mean([pq.specificity, pq.completeness, pq.ambiguity, pq.structure],'omitnan');

        is_ife(i) = strcmp(raw(i).source,'ifeval');
        if is_ife(i)
            Sx = s(getflag(sc,'verifiable') == 1);
            n_strict(i)       = numel(Sx);
            n_unverifiable(i) = sum(isnan(Sx));
            if ~isempty(Sx)
                sa_nan_as_fail(i) = mean(Sx == 1);       % what v1 computed
                ch = Sx(~isnan(Sx));
                if ~isempty(ch), sa_checked(i) = mean(ch == 1); end   % correct
            end
        end
    end

    % ---------------- 1. routing ----------------
    fprintf('\n--- 1. Confidence routing (computed in v1, never applied) ---\n');
    fprintf('Loose judgments: %d | flagged needs_review: %d (%.1f%%)\n', ...
        sum(n_loose), sum(n_rout), 100*sum(n_rout)/max(1,sum(n_loose)));
    fprintf('Items left with NO score once the filter is applied: %d of %d\n', ...
        sum(isnan(cal_kept)), n);

    % ---------------- 2. divergence ----------------
    fprintf('\n--- 2. Divergence, tie-corrected (manuscript used the no-ties formula) ---\n');
    show('amb x calibrated, UNFILTERED (v1)', pq_amb, cal_all);
    show('amb x calibrated, router applied', pq_amb, cal_kept);
    show('amb x RAW,        router applied', pq_amb, raw_kept);
    show('4-dim x calibrated, router applied', pq_c4, cal_kept);
    fprintf(['\nIf raw and calibrated differ materially, the length regression is\n' ...
             'shaping the result and must be reported both ways.\n']);

    % ---------------- 3. strict accuracy ----------------
    fprintf('\n--- 3. Strict accuracy on the IFEval subset (n = %d items) ---\n', sum(is_ife));
    if any(is_ife)
        fprintf('Strict constraints extracted: %d | unverifiable (NaN): %d (%.1f%%)\n', ...
            sum(n_strict), sum(n_unverifiable), ...
            100*sum(n_unverifiable)/max(1,sum(n_strict)));
        fprintf('v1 method  (NaN counted as FAILURE): %.4f\n', mean(sa_nan_as_fail,'omitnan'));
        fprintf('correct    (NaN excluded)          : %.4f\n', mean(sa_checked,'omitnan'));
        fprintf(['\nThe gap between those two lines is the artefact. Neither is the\n' ...
                 'real figure: both score LLM-invented constraints. verify_ifeval.py\n' ...
                 'against the official registry gives the number to publish.\n']);
    else
        fprintf('No IFEval items in this run.\n');
    end

    fprintf('\n===== END =====\n');
end

function show(label, x, y)
    [rho,p,ci,nn] = spearman_ties(x(:), y(:));
    fprintf('  %-36s rho = %+.4f  CI [%+.3f, %+.3f]  p = %.4f  n = %d\n', ...
        label, rho, ci(1), ci(2), p, nn);
end

function v = getnum(s, field)
    v = nan(1, numel(s));
    if isempty(s) || ~isfield(s, field), return; end
    for i = 1:numel(s)
        x = s(i).(field);
        if ~isempty(x) && isnumeric(x) && isscalar(x), v(i) = double(x); end
    end
end

function v = getflag(s, field)
    v = zeros(1, numel(s));
    if isempty(s) || ~isfield(s, field), return; end
    for i = 1:numel(s)
        x = s(i).(field);
        if ~isempty(x) && (islogical(x) || isnumeric(x)) && isscalar(x)
            v(i) = double(x ~= 0);
        end
    end
end
