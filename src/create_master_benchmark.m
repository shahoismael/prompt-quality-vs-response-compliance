function create_master_benchmark()
    [wildchat, lmsys, ifeval] = load_datasets();

    keywords = {'explain', 'learn', 'how to', 'what is', 'teach', 'understand'};

    naturalistic = [wildchat; lmsys];
    num_raw = length(naturalistic);
    nat_prompts = {};
    nat_source = {};
    nat_stratum = {};
    excluded_length = 0;
    stripped_name_tokens = 0;
    MAX_WORDS = 1000; % p90 cutoff observed in actual WildChat distribution;
                       % prompts beyond this are pasted documents, not
                       % authored instructions, and distort specificity scoring

    for i = 1:num_raw
        p = naturalistic(i).prompt;

        if length(strsplit(p)) > MAX_WORDS
            excluded_length = excluded_length + 1;
            continue; % excluded, not truncated: truncation would corrupt
                      % the constraint content mid-instruction
        end

        % Strip LMSYS anonymization artifacts (NAME_1, NAME_2, ...) before
        % any agent scores this text. Left unstripped, these tokens get
        % judged as ambiguous/meaningless content rather than recognized
        % as a redaction artifact of the source corpus.
        if ~isempty(regexp(p, 'NAME_\d+', 'once'))
            stripped_name_tokens = stripped_name_tokens + 1;
            p = regexprep(p, 'NAME_\d+', '[REDACTED]');
        end

        nat_prompts{end+1} = p; %#ok<AGROW>
        nat_source{end+1} = naturalistic(i).source; %#ok<AGROW>
        p_lower = lower(p);
        stratum = 'other';
        for k = 1:numel(keywords)
            if contains(p_lower, keywords{k})
                stratum = keywords{k};
                break;
            end
        end
        nat_stratum{end+1} = stratum; %#ok<AGROW>
    end
    num_nat = numel(nat_prompts);

    fprintf('Excluded %d/%d naturalistic prompts over %d words.\n', excluded_length, num_raw, MAX_WORDS);
    fprintf('Stripped NAME_n placeholders in %d prompts (replaced with [REDACTED]).\n', stripped_name_tokens);

    % IFEval prompts keep their OWN real constraints, no cyclic reuse onto
    % unrelated naturalistic prompts. This is the only subset with genuine
    % deterministic ground truth (see Methodology 3.2).
    num_ife = length(ifeval);
    ife_prompts = cell(1, num_ife);
    ife_source = cell(1, num_ife);
    ife_stratum = cell(1, num_ife);
    ife_ground_truth = cell(1, num_ife);
    for i = 1:num_ife
        ife_prompts{i} = ifeval(i).prompt;
        ife_source{i} = 'ifeval';
        ife_stratum{i} = 'verifiable';
        ife_ground_truth{i} = ifeval(i).instruction_id_list;
    end

    master_benchmark = struct();
    master_benchmark.prompts = [ife_prompts, nat_prompts];
    master_benchmark.source = [ife_source, nat_source];
    master_benchmark.stratum = [ife_stratum, nat_stratum];
    % Ground truth only populated for the ifeval subset. Naturalistic entries
    % get an empty placeholder, NOT a fabricated constraint. Accuracy metrics
    % must check source == 'ifeval' before using this field (see run_simulation.m).
    master_benchmark.ground_truth_constraints = [ife_ground_truth, cell(1, num_nat)];

    json_str = jsonencode(master_benchmark);
    if ~exist('data', 'dir'), mkdir('data'); end
    fid = fopen('data/master_benchmark.json', 'w');
    fprintf(fid, '%s', json_str);
    fclose(fid);

    fprintf('Created master_benchmark.json: %d IFEval (ground-truth eligible) + %d naturalistic prompts.\n', ...
        num_ife, num_nat);
    fprintf('NOTE: verify num_ife against your actual filtered ifeval_filtered.json count before using it as N_IFEval in the paper.\n');
end