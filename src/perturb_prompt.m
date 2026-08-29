function perturbed = perturb_prompt(prompt, variant)
% Content-preserving perturbations used for the judge-stability test.
%   'A' - typographical noise (adjacent-key substitution in 10% of words)
%   'B' - surface formatting only (doubled spacing, paragraph breaks)
%   'C' - no-op; verbosity is injected into the RESPONSE (perturb_response.m)
%
% FIX (v2): Variant B previously used strrep(p, '.', '.\n\n') inside single
% quotes. MATLAB single-quoted strings do not interpret escapes, so this
% inserted the two literal characters backslash-n rather than a newline;
% the API layer then escaped the backslash again and the model received
% literal "\\n\\n". Variant B therefore never tested formatting sensitivity.
% It now inserts real newlines.

    switch variant
        case 'A'
            words = strsplit(prompt);
            num_words = numel(words);
            num_typos = max(1, round(num_words * 0.1));
            keyboard_layout = 'qwertyuiopasdfghjklzxcvbnm';

            for i = 1:num_typos
                idx = randi(num_words);
                word = words{idx};
                if length(word) > 2
                    char_idx = randi(length(word) - 2) + 1;
                    target_char = lower(word(char_idx));
                    pos = strfind(keyboard_layout, target_char);
                    if ~isempty(pos)
                        adjacent_indices = [];
                        if pos > 1
                            adjacent_indices = [adjacent_indices, pos - 1]; %#ok<AGROW>
                        end
                        if pos < length(keyboard_layout)
                            adjacent_indices = [adjacent_indices, pos + 1]; %#ok<AGROW>
                        end
                        if ~isempty(adjacent_indices)
                            new_char = keyboard_layout(adjacent_indices(randi(numel(adjacent_indices))));
                            if word(char_idx) == upper(word(char_idx)) && ...
                                    word(char_idx) ~= lower(word(char_idx))
                                new_char = upper(new_char);
                            end
                            word(char_idx) = new_char;
                            words{idx} = word;
                        end
                    end
                end
            end
            perturbed = strjoin(words, ' ');

        case 'B'
            perturbed = strrep(prompt, ' ', '  ');
            perturbed = strrep(perturbed, '.', ['.' newline newline]);

        case 'C'
            perturbed = prompt;

        otherwise
            perturbed = prompt;
    end
end
