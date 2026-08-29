function main_simulation()
    clear; clc;
    
    % Ensure src is in path
    addpath('src');
    
    % Prompt user for input
    num_prompts = input('Enter number of prompts to simulate (default 10): ');
    if isempty(num_prompts) || ~isnumeric(num_prompts)
        num_prompts = 10;
        fprintf('Invalid input. Defaulting to 10 prompts.\n');
    end
    
    fprintf('\nStarting simulation for %d prompts...\n', num_prompts);
    
    % Run pipeline
    run_simulation(num_prompts);
    
    % Generate plots
    fprintf('\nGenerating plots...\n');
    plot_results();
    
    disp('Simulation complete. Press Enter to exit.');
    pause;
end