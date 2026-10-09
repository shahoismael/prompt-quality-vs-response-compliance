function export_factorial(in_name, out_name)
% Writes the factorial-ablation arms to CSV for the released results folder.
% Base MATLAB only; no toolboxes.
%
% RUN FROM THE PROJECT ROOT.
%   export_factorial('results/ablation_factorial.mat', ...
%                    'repo/results/ablation_factorial_120.csv')

    if nargin < 1 || isempty(in_name),  in_name  = 'results/ablation_factorial.mat'; end
    if nargin < 2 || isempty(out_name), out_name = 'repo/results/ablation_factorial_120.csv'; end
    assert(exist(in_name, 'file') == 2, ['Not found: ' in_name '. Run from the PROJECT ROOT.']);

    S = load(in_name);
    T = S.factorial_results;
    T = renamevars(T, 'src_index', 'main_run_idx');
    writetable(T, out_name);
    fprintf('Wrote %s (%d rows)\n', out_name, height(T));
    fprintf('NaN scores: naive3 %d | rub1 %d\n', ...
        sum(isnan(T.naive3_score)), sum(isnan(T.rub1_score)));
end
