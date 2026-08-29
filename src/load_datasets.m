function [wildchat_data, lmsys_data, ifeval_data] = load_datasets()
    wildchat_data = jsondecode(fileread('data/wildchat_filtered.json'));
    lmsys_data = jsondecode(fileread('data/lmsys_filtered.json'));
    ifeval_data = jsondecode(fileread('data/ifeval_filtered.json'));

    fprintf('Loaded %d WildChat prompts\n', length(wildchat_data));
    fprintf('Loaded %d LMSYS prompts\n', length(lmsys_data));
    fprintf('Loaded %d IFEval prompts\n', length(ifeval_data));
end