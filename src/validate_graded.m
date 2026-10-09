function validate_graded(out_name)
% Graded construct validation for the Prompt Quality Agent.
%
% The extreme-groups design in validate_construct.m deletes 60% of a prompt's
% words and shuffles the rest, which removes well-formedness rather than
% prompt quality. A measure responsive only to grammaticality would reach the
% same ceiling. This script holds well-formedness constant and varies only the
% number of constraints the prompt states.
%
% Method: take IFEval prompts, split into sentences, identify the sentences
% that carry a constraint, and build variants with k constraint sentences
% removed, k = 0, 1, 2. Every variant is a grammatical prompt. Score each
% with the unchanged PromptQualityAgent and test for a monotone response.
%
% RUN FROM THE PROJECT ROOT.
%   validate_graded('results/graded_validation.mat')

    if nargin < 1 || isempty(out_name)
        out_name = 'results/graded_validation.mat';
    end

    rng(42, 'twister');
    api = Ollama_API('qwen2.5:7b');

    % IFEval items ship with the repository (Apache-2.0); no rebuild needed.
    raw_ife = jsondecode(fileread('data/ifeval_filtered.json'));
    if isstruct(raw_ife)
        pool = arrayfun(@(x) char(string(x.prompt)), raw_ife, 'UniformOutput', false);
    else
        pool = cellfun(@(x) char(string(x.prompt)), raw_ife, 'UniformOutput', false);
    end
    pool = pool(:);

    fprintf('Loaded %d IFEval prompts.\n', numel(pool));

    % Keep prompts with at least three constraint sentences, so k = 0..2 all
    % exist and every variant still states at least one constraint. If too few
    % qualify, relax the threshold rather than returning an empty design.
    cand = {}; ncon_all = zeros(numel(pool),1); nsent_all = zeros(numel(pool),1);
    for i = 1:numel(pool)
        [sent, isc] = split_constraints(pool{i});
        ncon_all(i)  = sum(isc);
        nsent_all(i) = numel(sent);
    end
    fprintf('Constraint sentences per prompt: median %d, max %d.\n', ...
        median(ncon_all), max(ncon_all));
    thr = 3;
    while thr >= 1
        keepi = ncon_all >= thr & nsent_all >= thr;
        if sum(keepi) >= 30, break; end
        thr = thr - 1;
    end
    assert(thr >= 1 && sum(keepi) > 0, ...
        'No prompt yielded a constraint sentence. Check split_constraints.');
    if thr < 3
        fprintf('Relaxed threshold to %d constraint sentences (%d prompts).\n', thr, sum(keepi));
    end
    cand = pool(keepi);
    n_target = min(80, numel(cand));
    fprintf('Eligible prompts: %d. Using %d.\n', numel(cand), n_target);
    sel  = randperm(numel(cand), n_target);
    cand = cand(sel);

    K     = 0:2;
    Q     = nan(n_target, numel(K));   % composite (ambiguity + structure) / 2
    NCON  = nan(n_target, numel(K));   % constraint sentences remaining
    fprintf('Graded validation: %d prompts x %d strata...\n', n_target, numel(K));

    for i = 1:n_target
        [sent, isc] = split_constraints(cand{i});
        cidx = find(isc);
        cidx = cidx(randperm(numel(cidx)));   % fixed by the global seed
        for j = 1:numel(K)
            drop = cidx(1:K(j));
            keep = setdiff(1:numel(sent), drop);
            variant = strtrim(strjoin(sent(sort(keep)), ' '));
            try
                q = PromptQualityAgent(variant, api);
                Q(i,j)    = (q.ambiguity + q.structure) / 2;
                NCON(i,j) = sum(isc) - K(j);
            catch err
                fprintf('Prompt %d, k=%d error (%s). Skipping.\n', i, K(j), err.message);
            end
        end
        fprintf('Prompt %d/%d done\n', i, n_target);
    end

    if ~exist('results', 'dir'), mkdir('results'); end
    save(out_name, 'Q', 'NCON', 'K', 'cand', '-v7.3');

    fprintf('\n===== GRADED VALIDATION =====\n');
    for j = 1:numel(K)
        v = Q(:,j); v = v(~isnan(v));
        fprintf('k=%d removed: n=%3d  mean Q=%.3f  SD=%.3f\n', K(j), numel(v), mean(v), std(v));
    end

    % Adjacent-stratum paired tests: does dropping one more constraint lower Q?
    for j = 1:numel(K)-1
        ok = ~isnan(Q(:,j)) & ~isnan(Q(:,j+1));
        report_wilcoxon(sprintf('k=%d vs k=%d', K(j), K(j+1)), Q(ok,j), Q(ok,j+1));
        fprintf('   discrimination rate (fewer removed scores higher): %.3f\n', ...
            mean(Q(ok,j) > Q(ok,j+1)));
    end

    % Monotonicity across all item-variant pairs.
    ok = ~isnan(Q) & ~isnan(NCON);
    rho = spearman_ties(NCON(ok), Q(ok));
    fprintf('Spearman(constraints remaining, Q) = %+.3f over %d item-variants\n', rho, sum(ok(:)));
    fprintf('===== END =====\n');
end

function [sent, isc] = split_constraints(prompt)
% Sentence split, then flag sentences that state a constraint. The lexicon is
% derived from the six IFEval-derived constraint categories used in Eq. (1):
% length, format, keyword, case, structural, audience. Plain CONTAINS is used
% rather than a regular expression, so behaviour does not depend on how a
% MATLAB release handles inline mode modifiers.
    t = strtrim(prompt);
    t = regexprep(t, '([.!?])\s+', '$1<<S>>');
    s = strsplit(t, '<<S>>');
    s = s(~cellfun(@(x) isempty(strtrim(x)), s));
    if numel(s) < 2
        s = strsplit(t, {sprintf('\n'), '; '});
        s = s(~cellfun(@(x) isempty(strtrim(x)), s));
    end

    terms = { ...
        'word','sentence','paragraph','bullet','section','title','json', ...
        'format','markdown','lowercase','uppercase','capital','letter', ...
        'must','should','do not','don''t','avoid','include','exclude', ...
        'contain','at least','at most','exactly','no more than','fewer than', ...
        'wrap','quotation','postscript','placeholder','highlight','comma', ...
        'bold','italic','list','response language','entire','only','refrain'};

    isc = false(1, numel(s));
    for q = 1:numel(s)
        low = lower(s{q});
        for r = 1:numel(terms)
            if contains(low, terms{r})
                isc(q) = true;
                break;
            end
        end
    end
    sent = s;
end

function report_wilcoxon(label, a, b)
% Paired Wilcoxon signed-rank, normal approximation, base MATLAB only.
    d = a - b; d = d(d ~= 0); n = numel(d);
    if n < 10, fprintf('%s: insufficient pairs (n=%d)\n', label, n); return; end
    [~, sidx] = sort(abs(d));
    ranks = zeros(n,1); sa = abs(d(sidx)); i = 1;
    while i <= n
        j = i;
        while j < n && sa(j+1) == sa(i), j = j + 1; end
        ranks(sidx(i:j)) = (i + j) / 2; i = j + 1;
    end
    Wp = sum(ranks(d > 0)); Wm = sum(ranks(d < 0));
    mu = n*(n+1)/4; sd = sqrt(n*(n+1)*(2*n+1)/24);
    z = (min(Wp,Wm) - mu) / sd; p = erfc(abs(z)/sqrt(2));
    rb = (Wp - Wm) / (Wp + Wm);
    fprintf('%s: n=%d, z=%.2f, p=%.4g, rank-biserial r=%.3f\n', label, n, z, p, rb);
end
