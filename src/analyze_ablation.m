function analyze_ablation(sim_name, base_name)
% Computes every statistic for Results sections 4.5 and 4.6 from a v2 run.
% RUN FROM THE PROJECT ROOT. Paste the whole printed block back into chat.
%
%   analyze_ablation('results/simulation_results_main500_v2.mat', ...
%                    'results/baseline_results_v2.mat')
%
% v2 changes:
%   - reads the v2 table (Raw_Loose_Score, FixedRubric_Score, N_Routed, ...)
%   - reports MEDIANS alongside means for every latency and variance figure,
%     because both distributions are heavily right-skewed by model-reload
%     stalls and reporting means alone previously inverted the naive/CoT
%     latency ordering
%   - compares the baselines against the framework on the SAME item subset
%   - uses the tie-corrected Spearman

    if nargin < 1 || isempty(sim_name),  sim_name  = 'results/simulation_results.mat'; end
    if nargin < 2 || isempty(base_name), base_name = 'results/baseline_results.mat'; end

    S = load(sim_name);
    B = load(base_name);
    results = S.results;
    base    = B.baseline_results;

    % align framework rows to the baseline subset
    if isfield(B, 'subset_index')
        fw = results(B.subset_index, :);
    else
        fw = results;
    end

    fprintf('\n===== ABLATION ANALYSIS (v2) =====\n');
    fprintf('Framework rows: %d | Baseline rows: %d | aligned: %d\n', ...
        height(results), height(base), height(fw));
    fprintf('IFEval in full run: %d\n', sum(string(results.Source) == "ifeval"));

    % ---- 4.5 confidence routing (now actually applied upstream) ----
    fprintf('\n--- 4.5 Routing ---\n');
    fprintf('Loose judgments: %d | deferred to review: %d (%.1f%%)\n', ...
        sum(results.N_Loose), sum(results.N_Routed), ...
        100*sum(results.N_Routed)/max(1,sum(results.N_Loose)));
    fprintf('Items left with NO automated score: %d (%.1f%%)\n', ...
        sum(isnan(results.Calibrated_Loose_Score)), ...
        100*mean(isnan(results.Calibrated_Loose_Score)));

    % ---- accuracy ----
    fprintf('\n--- Accuracy (IFEval subset; definitive figures from verify_ifeval.py) ---\n');
    fprintf('Strict (checked only): %.4f | unverifiable constraints dropped: %d\n', ...
        mean(results.Strict_Acc_Checked,'omitnan'), sum(results.Strict_N_Unverifiable));
    fprintf('Loose: %.4f\n', mean(results.Loose_Acc,'omitnan'));

    % ---- system comparison ----
    fprintf('\n--- System comparison (aligned subset) ---\n');
    row('Framework (calibrated)', fw.Calibrated_Loose_Score, fw.Bias_Variance, fw.Latency_s);
    row('Framework (raw)',        fw.Raw_Loose_Score,        fw.Bias_Variance, fw.Latency_s);
    row('Baseline 1 naive',       base.naive_score,          base.naive_bv,    base.naive_lat_total);
    row('Baseline 2 CoT',         base.cot_score,            base.cot_bv,      base.cot_lat_total);

    fprintf('\n--- Latency, like for like (ONE judgment) ---\n');
    fprintf('naive: mean %.1f s, median %.1f s, max %.1f s\n', ...
        mean(base.naive_lat_one,'omitnan'), median(base.naive_lat_one,'omitnan'), max(base.naive_lat_one));
    fprintf('CoT  : mean %.1f s, median %.1f s, max %.1f s\n', ...
        mean(base.cot_lat_one,'omitnan'), median(base.cot_lat_one,'omitnan'), max(base.cot_lat_one));
    fprintf('NOTE: framework Latency_s is a WHOLE-ITEM total (generation + prompt\n');
    fprintf('      quality + four scoring passes + fixed-rubric arm). Do not divide\n');
    fprintf('      it by a single-judgment baseline figure and call it a speed ratio.\n');

    fprintf('\n--- Midpoint clustering (share in [2.5, 3.5]) ---\n');
    fprintf('Framework %.3f | naive %.3f | CoT %.3f\n', ...
        mid_share(fw.Calibrated_Loose_Score), mid_share(base.naive_score), mid_share(base.cot_score));
    fprintf('--- Ceiling clustering (share == 5) ---\n');
    fprintf('Framework %.3f | naive %.3f | CoT %.3f\n', ...
        ceil_share(fw.Calibrated_Loose_Score), ceil_share(base.naive_score), ceil_share(base.cot_score));

    fprintf('\n--- Wilcoxon signed-rank (paired, two-sided) ---\n');
    wilcoxon('Score: framework vs naive', fw.Calibrated_Loose_Score, base.naive_score);
    wilcoxon('Score: framework vs CoT',   fw.Calibrated_Loose_Score, base.cot_score);
    wilcoxon('BV:    framework vs naive', fw.Bias_Variance,          base.naive_bv);
    wilcoxon('BV:    framework vs CoT',   fw.Bias_Variance,          base.cot_bv);

    if isfield(S, 'cal_info') && S.cal_info.fitted
        c = S.cal_info;
        fprintf('\n--- Calibration fit ---\n');
        fprintf('beta1 = %.6f (SE %.6f), t = %.2f, R2 = %.4f, n = %d\n', ...
            c.beta1, c.se_beta1, c.beta1/max(eps,c.se_beta1), c.r2, c.n);
        if abs(c.beta1/max(eps,c.se_beta1)) < 1.96
            fprintf('beta1 is NOT significant: the length correction is doing nothing.\n');
            fprintf('Say so, or drop the Calibration Agent from the contribution claims.\n');
        end
    end

    if isfield(S, 'div')
        fprintf('\n--- Divergence (tie-corrected Spearman, full run) ---\n');
        f = fieldnames(S.div);
        for k = 1:numel(f)
            d = S.div.(f{k});
            fprintf('  %-18s rho = %+.4f  95%% CI [%+.3f, %+.3f]  p = %.4f  n = %d\n', ...
                f{k}, d.rho, d.ci(1), d.ci(2), d.p, d.n);
        end
        fprintf('\nDECISION RULE: amb_vs_fixed is the one that matters. Criteria in\n');
        fprintf('amb_vs_cal are extracted FROM the prompt, so that estimate is\n');
        fprintf('confounded by design. If amb_vs_fixed is also flat, the divergence\n');
        fprintf('claim survives. If it is positive, the null was the architecture.\n');
    end

    fprintf('\n===== END =====\n');
end

function row(label, score, bv, lat)
    fprintf('%-24s score %.3f (med %.3f) | BV %.3f (med %.3f) | lat %.1f s (med %.1f)\n', ...
        label, mean(score,'omitnan'), median(score,'omitnan'), ...
        mean(bv,'omitnan'), median(bv,'omitnan'), ...
        mean(lat,'omitnan'), median(lat,'omitnan'));
end

function s = mid_share(x)
    x = x(~isnan(x)); s = mean(x >= 2.5 & x <= 3.5);
end

function s = ceil_share(x)
    x = x(~isnan(x)); s = mean(x >= 4.999);
end

function wilcoxon(label, a, b)
    a = a(:); b = b(:);
    m = min(numel(a), numel(b));
    a = a(1:m); b = b(1:m);
    ok = ~isnan(a) & ~isnan(b);
    d = a(ok) - b(ok);
    d = d(d ~= 0);
    n = numel(d);
    if n < 10
        fprintf('%s: insufficient pairs (n=%d)\n', label, n); return;
    end
    [~, idx] = sort(abs(d));
    ranks = zeros(n,1); sa = abs(d(idx)); i = 1;
    while i <= n
        j = i;
        while j < n && sa(j+1) == sa(i), j = j + 1; end
        ranks(idx(i:j)) = (i + j) / 2;
        i = j + 1;
    end
    Wp = sum(ranks(d > 0)); Wm = sum(ranks(d < 0));
    mu = n*(n+1)/4; sd = sqrt(n*(n+1)*(2*n+1)/24);
    z = (min(Wp,Wm) - mu) / sd;
    p = erfc(abs(z)/sqrt(2));
    rb = (Wp - Wm) / (Wp + Wm);
    fprintf('%s: n=%d, z=%.2f, p=%.4g, rank-biserial r=%.3f, medians %.2f vs %.2f\n', ...
        label, n, z, p, rb, median(a(ok)), median(b(ok)));
end
