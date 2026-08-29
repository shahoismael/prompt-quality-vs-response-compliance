function make_figs_data(sim_name, base_name, out_dir)
% Figures 2, 3, 4 for the manuscript. RUN FROM THE PROJECT ROOT.
%
%   make_figs_data('results/simulation_results_main500_v2.mat', ...
%                  'results/baseline_results_v2.mat', ...
%                  'D:\claude_projects\R3\final-sub-25-30-aug-2026\figs')

    if nargin < 1 || isempty(sim_name),  sim_name  = 'results/simulation_results_main500_v2.mat'; end
    if nargin < 2 || isempty(base_name), base_name = 'results/baseline_results_v2.mat'; end
    if nargin < 3 || isempty(out_dir)
        out_dir = 'D:\claude_projects\R3\final-sub-25-30-aug-2026\figs';
    end
    if ~isfolder(out_dir), mkdir(out_dir); end

    S = load(sim_name);
    B = load(base_name);
    R = S.results;
    base = B.baseline_results;

    % ---- response screen (paper Section 3.2) --------------------------------
    % Two responses are repetition loops (24,271 and 33,864 words, type-token
    % ratio < 0.02). They determine the OLS calibration fit and must be removed
    % before anything derived from Calibrated_Loose_Score is plotted.
    % the generated text lives in raw_results, not in the results table
    txt = {S.raw_results.baseline_resp};
    L   = zeros(numel(txt),1); ttr = zeros(numel(txt),1);
    for i = 1:numel(txt)
        w = strsplit(strtrim(txt{i}));
        L(i)   = numel(w);
        ttr(i) = numel(unique(w)) / max(1, numel(w));
    end
    keepAll = ttr >= 0.10;

    % refit the length correction on screened data, then recompute S_cal
    ok  = keepAll & ~isnan(R.Raw_Loose_Score);
    bb  = [ones(sum(ok),1) L(ok)] \ R.Raw_Loose_Score(ok);
    % NOTE: MATLAB's two-argument min/max IGNORE NaN, so max(1,NaN) returns 1.
    % Clipping without restoring the missing values would turn every router-
    % deferred item into a spurious score of 1.
    Scal = min(5, max(1, R.Raw_Loose_Score - bb(2)*(L - mean(L(ok)))));
    Scal(isnan(R.Raw_Loose_Score)) = NaN;
    R.Calibrated_Loose_Score = Scal;
    fprintf('Screen: kept %d of %d | refit beta1 = %+.4e\n', ...
        sum(keepAll), numel(keepAll), bb(2));

    R    = R(keepAll, :);
    ksub = keepAll(B.subset_index);
    fw   = S.results(B.subset_index(ksub), :);
    fw.Calibrated_Loose_Score = Scal(B.subset_index(ksub));
    base = base(ksub, :);

    % Colourblind-safe (Okabe-Ito), greyscale-separable
    C.fw    = [0.00 0.45 0.70];
    C.naive = [0.90 0.62 0.00];
    C.cot   = [0.00 0.62 0.45];
    C.grey  = [0.45 0.45 0.45];
    FS = 9;

    Q   = R.Prompt_Ambiguity_Structure;
    Sfx = R.FixedRubric_Score;
    Scl = R.Calibrated_Loose_Score;

    %% ---------------- Figure 2: divergence, two panels ----------------
    f = newfig(17.8, 8.0);

    panel(1, Q, Sfx, 'Prompt-independent criteria', ...
        'Fixed-rubric compliance $S_{\mathrm{fixed}}$', C.fw, FS);
    panel(2, Q, Scl, 'Prompt-derived criteria', ...
        'Extracted-criteria compliance $S_{\mathrm{cal}}$', C.cot, FS);

    exportfig(f, out_dir, 'Figure2_divergence');

    %% ---------------- Figure 3: score distributions ----------------
    f = newfig(17.8, 7.0);
    edges = 0.75:0.5:5.25;
    sets  = {fw.Calibrated_Loose_Score, base.naive_score, base.cot_score};
    names = {'Ensembled framework', 'Naive single-pass', 'Chain-of-thought'};
    cols  = {C.fw, C.naive, C.cot};

    for k = 1:3
        subplot(1, 3, k);
        x = sets{k}; x = x(~isnan(x));
        histogram(x, edges, 'Normalization', 'probability', ...
            'FaceColor', cols{k}, 'EdgeColor', 'w', 'LineWidth', 0.6);
        hold on;
        yl = [0 0.65];
        plot([median(x) median(x)], yl, '--', 'Color', C.grey, 'LineWidth', 1.1);
        ylim(yl); xlim([0.5 5.5]); xticks(1:5);
        xlabel('Compliance score', 'FontSize', FS);
        if k == 1, ylabel('Proportion of items', 'FontSize', FS); end
        title(sprintf('%s\nceiling = %.1f%%', names{k}, 100*mean(x >= 4.999)), ...
            'FontSize', FS, 'FontWeight', 'normal');
        set(gca, 'FontSize', FS, 'Box', 'off', 'TickDir', 'out', 'Layer', 'top');
        hold off;
    end
    exportfig(f, out_dir, 'Figure3_score_distributions');

    %% ---------------- Figure 4: bias variance + ceiling ----------------
    % A log axis would silently drop every item whose bias variance is exactly
    % zero, and those items are 38 to 61 percent of each system. Panel A uses a
    % cumulative distribution so the mass at zero is the visible y-intercept.
    f = newfig(17.8, 7.8);

    bvs = {fw.Bias_Variance, base.naive_bv, base.cot_bv};
    % single-line labels: MATLAB splits an embedded newline across ticks
    lab = {'Framework', 'Naive', 'CoT'};
    leg = {'Ensembled framework', 'Naive single-pass', 'Chain-of-thought'};

    % ---- Panel A: empirical CDF of bias variance ----
    subplot(1, 2, 1);
    hold on;
    h = gobjects(1,3);
    for k = 1:3
        v = sort(bvs{k}(~isnan(bvs{k})));
        y = (1:numel(v))' / numel(v);
        h(k) = stairs([0; v], [0; y], 'Color', cols{k}, 'LineWidth', 1.6);
        plot(0, mean(v == 0), 'o', 'Color', cols{k}, 'MarkerFaceColor', cols{k}, ...
            'MarkerSize', 5);
    end
    xlim([-0.02 0.55]); ylim([0 1.02]);
    xlabel('Bias variance', 'FontSize', FS);
    ylabel('Cumulative proportion of items', 'FontSize', FS);
    legend(h, leg, 'Location', 'southeast', 'Box', 'off', 'FontSize', FS-1.5);
    ttl = title('A   Stability under perturbation', 'FontSize', FS, ...
        'FontWeight', 'normal');
    set(ttl, 'Units', 'normalized', 'Position', [0 1.02 0], ...
        'HorizontalAlignment', 'left');
    set(gca, 'FontSize', FS, 'Box', 'off', 'TickDir', 'out');
    hold off;

    % ---- Panel B: exact stability against ceiling clustering ----
    subplot(1, 2, 2);
    hold on;
    w = 0.34;
    zsh = cellfun(@(x) mean(x(~isnan(x)) == 0), bvs);
    csh = cellfun(@(x) mean(x(~isnan(x)) >= 4.999), sets);
    for k = 1:3
        bar(k - w/2, zsh(k), w, 'FaceColor', cols{k}, 'EdgeColor', 'none');
        bar(k + w/2, csh(k), w, 'FaceColor', cols{k}, 'FaceAlpha', 0.30, ...
            'EdgeColor', cols{k}, 'LineWidth', 1.2);
        text(k - w/2, zsh(k) + 0.018, sprintf('%.2f', zsh(k)), ...
            'HorizontalAlignment', 'center', 'FontSize', FS-1.5);
        text(k + w/2, csh(k) + 0.018, sprintf('%.2f', csh(k)), ...
            'HorizontalAlignment', 'center', 'FontSize', FS-1.5);
    end
    p1 = patch('XData', NaN, 'YData', NaN, 'FaceColor', C.grey, 'EdgeColor', 'none');
    p2 = patch('XData', NaN, 'YData', NaN, 'FaceColor', C.grey, 'FaceAlpha', 0.30, ...
        'EdgeColor', C.grey, 'LineWidth', 1.2);
    legend([p1 p2], {'Bias variance = 0', 'Score at scale ceiling'}, ...
        'Location', 'northwest', 'Box', 'off', 'FontSize', FS-1.5);
    xlim([0.5 3.5]); ylim([0 0.88]); xticks(1:3); xticklabels(lab);
    ylabel('Share of items', 'FontSize', FS);
    ttl = title('B   Stability versus insensitivity', 'FontSize', FS, ...
        'FontWeight', 'normal');
    set(ttl, 'Units', 'normalized', 'Position', [0 1.02 0], ...
        'HorizontalAlignment', 'left');
    set(gca, 'FontSize', FS, 'Box', 'off', 'TickDir', 'out');
    hold off;

    exportfig(f, out_dir, 'Figure4_bias_variance');

    fprintf('Figures 2-4 written to %s\n', out_dir);
end

% ===================== helpers =====================

function f = newfig(w_cm, h_cm)
    f = figure('Units', 'centimeters', 'Position', [2 2 w_cm h_cm], ...
        'Color', 'w', 'PaperUnits', 'centimeters', ...
        'PaperSize', [w_cm h_cm], 'PaperPosition', [0 0 w_cm h_cm]);
end

function panel(k, x, y, ttl, ylab, col, FS)
    subplot(1, 2, k);
    ok = ~isnan(x) & ~isnan(y);
    x = x(ok); y = y(ok);

    jx = x + (rand(numel(x),1) - 0.5) * 0.10;
    jy = y + (rand(numel(y),1) - 0.5) * 0.06;
    scatter(jx, jy, 11, col, 'filled', 'MarkerFaceAlpha', 0.30, ...
        'MarkerEdgeColor', 'none');
    hold on;

    b = [ones(numel(x),1) x] \ y;
    xg = linspace(min(x), max(x), 50);
    plot(xg, b(1) + b(2)*xg, '-', 'Color', col, 'LineWidth', 1.8);

    rho = spearman_ties(x, y);
    text(0.04, 0.95, sprintf('\\rho = %+.3f', rho), 'Units', 'normalized', ...
        'FontSize', FS, 'VerticalAlignment', 'top');
    text(0.04, 0.87, sprintf('n = %d', numel(x)), 'Units', 'normalized', ...
        'FontSize', FS-1, 'VerticalAlignment', 'top', 'Color', [0.35 0.35 0.35]);

    xlim([0.8 5.2]); ylim([0.8 5.2]); xticks(1:5); yticks(1:5);
    xlabel('Prompt quality $Q$', 'Interpreter', 'latex', 'FontSize', FS);
    ylabel(ylab, 'Interpreter', 'latex', 'FontSize', FS);
    title(ttl, 'FontSize', FS, 'FontWeight', 'normal');
    set(gca, 'FontSize', FS, 'Box', 'off', 'TickDir', 'out', 'Layer', 'top');
    axis square;
    hold off;
end

function q = quartiles(v)
% [Q1 Q2 Q3] without the Statistics Toolbox (linear interpolation).
    v = sort(v(~isnan(v)));
    n = numel(v);
    q = zeros(1,3);
    p = [0.25 0.50 0.75];
    for i = 1:3
        h = (n - 1) * p(i) + 1;
        lo = floor(h); hi = ceil(h);
        q(i) = v(lo) + (h - lo) * (v(hi) - v(lo));
    end
end

function exportfig(f, out_dir, name)
    exportgraphics(f, fullfile(out_dir, [name '.tif']), 'Resolution', 600);
    exportgraphics(f, fullfile(out_dir, [name '.png']), 'Resolution', 600);
    exportgraphics(f, fullfile(out_dir, [name '.pdf']), 'ContentType', 'vector');
    close(f);
end
