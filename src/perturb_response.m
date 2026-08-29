function perturbed_resp = perturb_response(response)
% Appends verbose, semantically empty filler to a response to test whether
% the Semantic Scorer's loose scores shift with length alone, holding
% content and constraints fixed. This replaces perturb_prompt.m Variant C
% for verbosity-bias testing specifically: that variant lengthened the
% PROMPT, which does not test response verbosity bias at all.
    filler = [' To elaborate further and provide additional context, it is worth ' ...
        'noting that this topic involves several interconnected considerations ' ...
        'worth exploring in greater depth, as a more comprehensive treatment ' ...
        'often benefits from additional detail and thorough explanation.'];
    perturbed_resp = [response, filler];
end