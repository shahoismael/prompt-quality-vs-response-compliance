function plot_results()
    loaded = load('results/simulation_results.mat', 'results', 'divergence_rho');
    results = loaded.results;

    ifeval_rows = strcmp(results.Source, 'ifeval');
    nat_rows = ~ifeval_rows;

    % --- Plot 1: Strict/Loose Accuracy, IFEval subset only ---
    if any(ifeval_rows)
        figure('Position', [100, 100, 600, 400]);
        strict_mean = mean(results.Strict_Acc(ifeval_rows), 'omitnan');
        loose_mean = mean(results.Loose_Acc(ifeval_rows), 'omitnan');
        bar([strict_mean, loose_mean]);
        set(gca, 'XTickLabel', {'Strict Accuracy', 'Loose Accuracy'});
        title(sprintf('Ground-Truth Accuracy (IFEval subset, N=%d)', sum(ifeval_rows)));
        ylabel('Mean Accuracy');
        grid on;
        fprintf('IFEval Strict Accuracy: %.3f\n', strict_mean);
        fprintf('IFEval Loose Accuracy: %.3f\n', loose_mean);
    else
        fprintf('No IFEval-sourced rows in this run; skipping ground-truth accuracy plot.\n');
    end

    % --- Plot 2: Bias Variance by source (judge stability, full benchmark) ---
    figure('Position', [100, 550, 600, 400]);
    sources = unique(results.Source);
    bv_means = arrayfun(@(s) mean(results.Bias_Variance(strcmp(results.Source, s{1})), 'omitnan'), sources);
    bar(bv_means);
    set(gca, 'XTickLabel', sources);
    title('Judge Score Variance Under Perturbation, by Source');
    ylabel('Bias Variance');
    grid on;

    % --- Plot 3: Prompt Response Divergence scatter ---
    valid = ~isnan(results.Prompt_Ambiguity_Structure) & ~isnan(results.Calibrated_Loose_Score);
    if sum(valid) >= 5
        figure('Position', [750, 100, 600, 400]);
        scatter(results.Prompt_Ambiguity_Structure(valid), results.Calibrated_Loose_Score(valid), 40, 'filled');
        xlabel('Prompt Quality Score (ambiguity/structure)');
        ylabel('Calibrated Response Compliance Score');
        title(sprintf('Prompt Response Divergence (Spearman rho = %.3f)', loaded.divergence_rho));
        grid on;
        fprintf('Prompt Response Divergence (Spearman rho): %.3f\n', loaded.divergence_rho);
    else
        fprintf('Not enough valid rows to plot Prompt Response Divergence.\n');
    end

    % --- Summary printout ---
    fprintf('\n--- Summary ---\n');
    fprintf('Total rows: %d (IFEval: %d, Naturalistic: %d)\n', height(results), sum(ifeval_rows), sum(nat_rows));
    for i = 1:numel(sources)
        n = sum(strcmp(results.Source, sources{i}));
        fprintf('  %s: %d rows\n', sources{i}, n);
    end
end