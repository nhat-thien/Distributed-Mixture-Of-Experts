function red = param_MSE(estimated_mixture, true_mixture)

    [d,~] = size(true_mixture.experts);

    Beta   = true_mixture.experts(:);
    Alpha  = true_mixture.gates(:);
    sigma2 = true_mixture.variances(:);

    Beta_est   = estimated_mixture.experts(:);
    Alpha_est  = estimated_mixture.gates(:);
    sigma2_est = estimated_mixture.variances(:);
    
    
    param     = [Beta; sigma2; Alpha];
    param_est = [Beta_est; sigma2_est; Alpha_est];
    
    red = sum((param - param_est).^2)/(length(param)-d);

end