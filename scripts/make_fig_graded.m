function make_fig_graded(in_name, out_dir)
% Figure 4: graded construct validation. RUN FROM THE PROJECT ROOT.
%
%   make_fig_graded('results/graded_validation.mat', ...
%                   'D:\claude_projects\R3\1-Natural Language-Processing-(Cambridge)\figs')
%
% Two panels on one row.
%   A  Q by constraint-removal stratum, paired lines behind group means with
%      95% CI. The flat trajectory is the result.
%   B  the manipulation check: the deterministic dimensions, which are defined
%      over constraint content, do fall across the same strata.
%
% Panel B needs check_graded_manipulation to have been run with its SP and CP
% saved; if results/graded_manipulation.mat is absent the panel is drawn from
% the figures printed by that script, which are hard-coded below and marked.

    if nargin < 1 || isempty(in_name), in_name = 'results/graded_validation.mat'; end
    if nargin < 2 || isempty(out_dir)
        out_dir = 'D:\claude_projects\R3\1-Natural Language-Processing-(Cambridge)\figs';
    end
    if ~isfolder(out_dir), mkdir(out_dir); end

    G = load(in_name);           % Q (n x 3), NCON, K
    Q = G.Q; K = G.K;
    nK = numel(K);

    BLUE = [0.00 0.45 0.70];
    GREY = [0.45 0.45 0.45];
    ORNG = [0.90 0.62 0.00];
    GREEN= [0.00 0.62 0.45];
    FS = 9;

    f = figure('Units','centimeters','Position',[2 2 17.4 7.2], ...
               'Color','w','PaperPositionMode','auto');
    set(f,'DefaultAxesFontSize',FS,'DefaultTextFontSize',FS, ...
          'DefaultAxesFontName','Arial','DefaultTextFontName','Arial', ...
          'DefaultAxesLineWidth',0.5,'DefaultLineLineWidth',0.75);

    % ---- Panel A -----------------------------------------------------------
    ax = subplot(1,2,1); hold(ax,'on');
    for i = 1:size(Q,1)
        if all(~isnan(Q(i,:)))
            plot(K, Q(i,:), '-', 'Color', [GREY 0.18], 'LineWidth', 0.4);
        end
    end
    m = nan(1,nK); lo = nan(1,nK); hi = nan(1,nK);
    for j = 1:nK
        v = Q(:,j); v = v(~isnan(v));
        m(j) = mean(v);
        se   = std(v)/sqrt(numel(v));
        lo(j) = m(j) - 1.96*se; hi(j) = m(j) + 1.96*se;
    end
    errorbar(K, m, m-lo, hi-m, 'o-', 'Color', BLUE, 'MarkerFaceColor', BLUE, ...
             'MarkerSize', 5, 'LineWidth', 1.4, 'CapSize', 5);
    for j = 1:nK
        text(K(j), m(j)+0.42, sprintf('%.2f', m(j)), 'Color', BLUE, ...
             'HorizontalAlignment','center','FontSize',FS-1);
    end
    xlim([-0.35 nK-1+0.35]); ylim([1 5.4]);
    set(ax,'XTick',K,'YTick',1:5,'Box','off','TickDir','out','Layer','top');
    xlabel('Constraint sentences removed'); ylabel('Prompt quality  Q');
    title('A   Q does not track specification','FontWeight','normal', ...
          'HorizontalAlignment','left','Units','normalized','Position',[0 1.02 0]);
    rho = spearman_ties(G.NCON(~isnan(G.Q)), G.Q(~isnan(G.Q)));
    text(0.97, 0.06, sprintf('\\rho = %+.3f,  240 item-variants', rho), ...
         'Units','normalized','HorizontalAlignment','right','FontSize',FS-1, ...
         'Color',GREY);

    % ---- Panel B -----------------------------------------------------------
    ax = subplot(1,2,2); hold(ax,'on');
    if isfile('results/graded_manipulation.mat')
        M  = load('results/graded_manipulation.mat');
        sp = mean(M.SP,1,'omitnan'); cp = mean(M.CP,1,'omitnan');
        src = '';
    else
        sp = [0.545 0.436 0.312];     % from check_graded_manipulation
        cp = [0.338 0.275 0.198];
        src = '  (manipulation check)';
    end
    plot(K, sp, 's-', 'Color', ORNG, 'MarkerFaceColor', ORNG, 'MarkerSize', 5, 'LineWidth', 1.4);
    plot(K, cp, '^-', 'Color', GREEN, 'MarkerFaceColor', GREEN, 'MarkerSize', 5, 'LineWidth', 1.4);
    xlim([-0.35 nK-1+0.35]); ylim([0 0.72]);
    set(ax,'XTick',K,'YTick',0:0.2:0.6,'Box','off','TickDir','out','Layer','top');
    xlabel('Constraint sentences removed'); ylabel('Deterministic dimension');
    legend({'Specificity  S_p','Completeness  C_p'}, 'Location','northeast', ...
           'Box','off','FontSize',FS-1);
    title(['B   the manipulation did remove constraints' src],'FontWeight','normal', ...
          'HorizontalAlignment','left','Units','normalized','Position',[0 1.02 0]);

    exportgraphics(f, fullfile(out_dir,'HassenFig4.tif'), 'Resolution', 600);
    exportgraphics(f, fullfile(out_dir,'HassenFig4.png'), 'Resolution', 600);
    exportgraphics(f, fullfile(out_dir,'HassenFig4.pdf'), 'ContentType', 'vector');
    close(f);
    fprintf('Figure 4 written to %s\n', out_dir);
end
