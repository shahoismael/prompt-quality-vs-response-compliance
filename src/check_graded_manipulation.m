function check_graded_manipulation()
% Manipulation check for validate_graded. No model calls; runs in seconds.
% Regenerates the same variants under the same seed and reports the
% DETERMINISTIC prompt dimensions (specificity, completeness), which are
% defined over constraint content. If these track k and the model-judged
% dimensions do not, the graded test is mis-specified for Q = (A+T)/2 rather
% than evidence that Q lacks resolution.
%
%   check_graded_manipulation

    rng(42, 'twister');
    raw_ife = jsondecode(fileread('data/ifeval_filtered.json'));
    if isstruct(raw_ife)
        pool = arrayfun(@(x) char(string(x.prompt)), raw_ife, 'UniformOutput', false);
    else
        pool = cellfun(@(x) char(string(x.prompt)), raw_ife, 'UniformOutput', false);
    end
    pool = pool(:);

    ncon_all = zeros(numel(pool),1); nsent_all = zeros(numel(pool),1);
    for i = 1:numel(pool)
        [sent, isc] = split_constraints(pool{i});
        ncon_all(i) = sum(isc); nsent_all(i) = numel(sent);
    end
    thr = 3;
    while thr >= 1
        keepi = ncon_all >= thr & nsent_all >= thr;
        if sum(keepi) >= 30, break; end
        thr = thr - 1;
    end
    cand = pool(keepi);
    n_target = min(80, numel(cand));
    sel = randperm(numel(cand), n_target);
    cand = cand(sel);

    K = 0:2;
    SP = nan(n_target, numel(K)); CP = nan(n_target, numel(K));
    WORDS = nan(n_target, numel(K));
    for i = 1:n_target
        [sent, isc] = split_constraints(cand{i});
        cidx = find(isc); cidx = cidx(randperm(numel(cidx)));
        for j = 1:numel(K)
            drop = cidx(1:K(j));
            keep = setdiff(1:numel(sent), drop);
            variant = strtrim(strjoin(sent(sort(keep)), ' '));
            d = deterministic_dims(variant);
            SP(i,j) = d.specificity; CP(i,j) = d.completeness;
            WORDS(i,j) = numel(strsplit(variant));
        end
    end

    if ~exist('results', 'dir'), mkdir('results'); end
    save('results/graded_manipulation.mat', 'SP', 'CP', 'WORDS', 'K', '-v7.3');

    fprintf('\n===== MANIPULATION CHECK (deterministic dimensions) =====\n');
    for j = 1:numel(K)
        fprintf('k=%d: specificity %.3f  completeness %.3f  words %.0f\n', ...
            K(j), mean(SP(:,j)), mean(CP(:,j)), mean(WORDS(:,j)));
    end
    fprintf('Completeness fell in %.1f%% of prompts from k=0 to k=2\n', 100*mean(CP(:,3) < CP(:,1)));
    fprintf('Specificity  fell in %.1f%% of prompts from k=0 to k=2\n', 100*mean(SP(:,3) < SP(:,1)));
    fprintf('Mean words removed, k=0 to k=2: %.1f\n', mean(WORDS(:,1)-WORDS(:,3)));
    fprintf('===== END =====\n');
end

function q = deterministic_dims(prompt)
% Equations (1) and (2), copied from PromptQualityAgent so this runs standalone.
    checklist = struct( ...
        'length',    {{'word','sentence','paragraph','character','words','sentences','paragraphs'}}, ...
        'format',    {{'bullet','list','table','json','markdown','heading','numbered'}}, ...
        'keyword',   {{'include','must contain','mention','avoid','do not use','without using'}}, ...
        'case',      {{'uppercase','lowercase','capitalize','all caps'}}, ...
        'structure', {{'introduction','conclusion','start with','end with','begin with'}}, ...
        'audience',  {{'for a','audience','beginner','expert','child','professional'}});
    fields = fieldnames(checklist); p = lower(prompt); hits = 0;
    for i = 1:numel(fields)
        terms = checklist.(fields{i});
        for j = 1:numel(terms)
            if contains(p, terms{j}), hits = hits + 1; break; end
        end
    end
    q.completeness = hits / numel(fields);
    vague = {'something','stuff','good','nice','anything','whatever','some'};
    vh = 0;
    for i = 1:numel(vague), if contains(p, vague{i}), vh = vh + 1; end, end
    wc = numel(strsplit(prompt));
    q.specificity = max(0, min(1, (hits/numel(fields)) - vh*0.15 + min(0.3, wc/200)));
end

function [sent, isc] = split_constraints(prompt)
    t = strtrim(prompt);
    t = regexprep(t, '([.!?])\s+', '$1<<S>>');
    s = strsplit(t, '<<S>>');
    s = s(~cellfun(@(x) isempty(strtrim(x)), s));
    if numel(s) < 2
        s = strsplit(t, {sprintf('\n'), '; '});
        s = s(~cellfun(@(x) isempty(strtrim(x)), s));
    end
    terms = {'word','sentence','paragraph','bullet','section','title','json', ...
        'format','markdown','lowercase','uppercase','capital','letter', ...
        'must','should','do not','don''t','avoid','include','exclude', ...
        'contain','at least','at most','exactly','no more than','fewer than', ...
        'wrap','quotation','postscript','placeholder','highlight','comma', ...
        'bold','italic','list','response language','entire','only','refrain'};
    isc = false(1, numel(s));
    for q = 1:numel(s)
        low = lower(s{q});
        for r = 1:numel(terms)
            if contains(low, terms{r}), isc(q) = true; break; end
        end
    end
    sent = s;
end
