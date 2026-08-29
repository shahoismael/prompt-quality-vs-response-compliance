function [calibrated, cal_info] = CalibrationAgent(scored_constraints, response_text, all_responses, all_scores)
% Length-controlled calibration (Dubois et al., 2024 logic). Removes the
% length-attributable component of the raw loose score. Position bias is not
% corrected because this architecture is pointwise: one response is scored in
% isolation, so there is no second output whose position could be swapped.
%
% FIX (v2): the regression fit is now returned in cal_info so beta1, its
% standard error, and R-squared can be reported. Previously the slope was
% computed, applied, and discarded, which left the paper unable to state
% whether the correction did anything at all.
%
% cal_info fields: fitted (logical), beta0, beta1, se_beta1, r2, n, mean_length

    calibrated = scored_constraints;
    cal_info = struct('fitted', false, 'beta0', NaN, 'beta1', NaN, ...
        'se_beta1', NaN, 'r2', NaN, 'n', 0, 'mean_length', NaN);
    if isempty(calibrated), return; end

    response_length = numel(strsplit(response_text));

    if nargin >= 4 && ~isempty(all_responses) && numel(all_responses) >= 10
        lengths = cellfun(@(r) numel(strsplit(r)), all_responses);
        scores  = all_scores;
        valid   = ~isnan(scores);

        if sum(valid) >= 10
            X = [ones(sum(valid),1), lengths(valid)'];
            y = scores(valid)';
            beta = X \ y;

            resid = y - X*beta;
            dof   = numel(y) - 2;
            sigma2 = (resid' * resid) / max(1, dof);
            C = sigma2 * inv(X' * X); %#ok<MINV>
            ss_tot = sum((y - mean(y)).^2);
            ss_res = resid' * resid;

            cal_info.fitted      = true;
            cal_info.beta0       = beta(1);
            cal_info.beta1       = beta(2);
            cal_info.se_beta1    = sqrt(C(2,2));
            cal_info.r2          = 1 - ss_res / max(eps, ss_tot);
            cal_info.n           = numel(y);
            cal_info.mean_length = mean(lengths(valid));

            for i = 1:numel(calibrated)
                if calibrated(i).verifiable == false && ~isnan(calibrated(i).score)
                    length_attributable = beta(2) * (response_length - cal_info.mean_length);
                    calibrated(i).score = calibrated(i).score - length_attributable;
                    calibrated(i).score = max(1, min(5, calibrated(i).score));
                    calibrated(i).regression_fitted = true;
                end
            end
            return;
        end
    end

    % Batch too small for a reliable fit. Flag rather than silently adjust.
    for i = 1:numel(calibrated)
        if calibrated(i).verifiable == false
            calibrated(i).regression_fitted = false;
        end
    end
end
