function [strict_acc, loose_acc, raw_loose_score] = evaluate_prompt(prompt, response, extracted_constraints, api)
    scored = SemanticScorer(prompt, response, extracted_constraints, api);
    
    strict_scores = [scored([scored.verifiable] == 1).score];
    if isempty(strict_scores)
        strict_acc = NaN; 
    else
        strict_acc = mean(strict_scores == 1);
    end
    
    loose_scores = [scored([scored.verifiable] == 0).score];
    if isempty(loose_scores)
        loose_acc = NaN;
        raw_loose_score = NaN;
    else
        raw_loose_score = mean(loose_scores);
        loose_acc = mean(abs(loose_scores - 3) <= 1); 
    end
end