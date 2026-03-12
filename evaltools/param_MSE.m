function red = param_MSE(estimated_MoE, true_MoE)

    [d,~] = size(true_MoE.experts);

    Beta   = true_MoE.experts(:);
    Alpha  = true_MoE.gates(:);
    sigma2 = true_MoE.variances(:);

    Beta_est   = estimated_MoE.experts(:);
    Alpha_est  = estimated_MoE.gates(:);
    sigma2_est = estimated_MoE.variances(:);
    
    
    param     = [Beta; sigma2; Alpha];
    param_est = [Beta_est; sigma2_est; Alpha_est];
    
    red = sum((param - param_est).^2)/(length(param)-d);

end