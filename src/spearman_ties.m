function [rho, p, ci, n] = spearman_ties(x, y)
% Tie-corrected Spearman rank correlation, with a Fisher-transform CI.
% Base MATLAB only; no Statistics Toolbox required.
%
% FIX (v2): the previous implementation used the simplified sum-of-squared-
% rank-differences identity
%       rho = 1 - 6*sum(d^2) / (n*(n^2-1))
% which is exact ONLY when there are no ties. The prompt-quality variable
% takes roughly two dozen distinct values across several hundred items, so
% ties are pervasive and that identity is biased. On the previous main run
% it returned +0.013 where the correct tie-corrected value is -0.054.
%
% The correct general definition of Spearman's rho is the PEARSON
% correlation of the (tie-averaged) ranks, which is what this computes.

    ok = ~isnan(x) & ~isnan(y);
    x = x(ok); y = y(ok);
    n = numel(x);
    rho = NaN; p = NaN; ci = [NaN NaN];
    if n < 5, return; end

    rx = tied_rank(x(:));
    ry = tied_rank(y(:));

    rx = rx - mean(rx);
    ry = ry - mean(ry);
    denom = sqrt(sum(rx.^2) * sum(ry.^2));
    if denom == 0, return; end
    rho = sum(rx .* ry) / denom;

    % Fisher z with the standard Spearman variance inflation factor 1.06
    if abs(rho) < 1 && n > 3
        z  = atanh(rho);
        se = 1.06 / sqrt(n - 3);
        ci = tanh([z - 1.96*se, z + 1.96*se]);
        p  = erfc(abs(z / se) / sqrt(2));   % two-sided
    end
end

function r = tied_rank(v)
    [sorted_v, idx] = sort(v);
    n = numel(v);
    ranks = zeros(n, 1);
    i = 1;
    while i <= n
        j = i;
        while j < n && sorted_v(j+1) == sorted_v(i)
            j = j + 1;
        end
        ranks(idx(i:j)) = (i + j) / 2;
        i = j + 1;
    end
    r = ranks;
end
