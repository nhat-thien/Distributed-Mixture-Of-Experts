function [metrics, prediction] = compute_metrics(fit, true_mixture, X, Y, true_labels, verbose)

    N = size(X,1);
    K = length(true_mixture.variances);
    
    %======================================================================
    % FOR GLOBAL, DISTRIBUTED, MEDIAN AND AVERAGED ME MODELS
    % 1. Learning times
    % 2. Transportation distance
    % 3. Log likelihood on X_
    % 4. MSE between estimated and true parameters
    % 5. Relative Prediction Error, Correlation, RI, ARI, Clusstering Error
    %======================================================================
    
    if isfield(fit, 'gates')

        %------------------------------------------------------------------
        %  COMPUTE TRANSPORTATION DISTANCE
        %------------------------------------------------------------------
        if isfield(fit, 'reduced_mixture')
            % For Global, Median and Average MoE
            [plan, distance_matrix] = argmin_transportation_plan(fit.reduced_mixture, true_mixture, [ones(N,1) X]);
        else 
            % For Distributed MoE
            [plan, distance_matrix] = argmin_transportation_plan(fit, true_mixture, [ones(N,1) X]);
        end
        
        trandis       = sum(sum(sum(plan .* distance_matrix)));
        learning_time = fit.learning_time;
        mse_param     =  param_MSE(fit, true_mixture);
        

        %------------------------------------------------------------------
        %  PREDICTION
        %------------------------------------------------------------------

        estimated_Alpha  = fit.param.Alpha;
        estimated_alpha0 = fit.param.alpha0;
        estimated_Beta   = fit.param.Beta;
        estimated_beta0  = fit.param.beta0;
        estimated_sigma2 = fit.param.sigma2;

        H       = estimated_alpha0  +  X * estimated_Alpha;
        maxm    = max(H, [], 2);
        H       = H  -  maxm * ones(1,K);
        gatingProb = exp(H) ./ (sum(exp(H),2)*ones(1,K));

        % Predict the responses
        Y_K     = estimated_beta0  +  X * estimated_Beta;
        Y_pred = sum(Y_K .* gatingProb, 2);
        pred_labels      = MAP(gatingProb);
        
        %------------------------------------------------------------------
        %  COMPUTE LOGLIK
        %------------------------------------------------------------------
        log_Phi_xy = zeros(N,K);
        for k = 1:K
            log_Phi_y =  -0.5*log(2*pi) - 0.5*log(estimated_sigma2(k)) -0.5*((Y - estimated_beta0(k)*ones(N,1) - X*estimated_Beta(:,k)).^2)/estimated_sigma2(k);
            log_Phi_xy(:,k) = log(gatingProb(:,k)) + log_Phi_y;
        end
        log_sum_PhixPhiy = logsumexp(log_Phi_xy,2);
        loglik           = sum(log_sum_PhixPhiy);

        %------------------------------------------------------------------
        %  COMPUTE CRITERIA
        %------------------------------------------------------------------
        % Relative Predictions Error (RPE)
        RPE_   = RPE(Y, Y_pred);
        corr_  = corr(Y ,Y_pred);
        [ARI_, RI_]  = RandIndex(true_labels, pred_labels);
        ClustErr_  = clusteringError(true_labels, pred_labels);

        
    %======================================================================
    % FOR MIXTURE OF LINEAR REGRESSION MODEL
    % No gating functions
    % Do not compute transporttation distance, MSE_param
    %======================================================================
    else
        
        trandis       = NaN;
        learning_time = fit.learning_time;
        mse_param     = NaN;
        
        estimated_Beta    = fit.param.Beta;
        estimated_beta0   = fit.param.beta0;
        estimated_sigma2  = fit.param.sigma2;
        estimated_weights = fit.param.weights;

        % -----------------------------------------------------------------
        %  PREDICTING
        % -----------------------------------------------------------------
        log_Phi_xy = zeros(N,K);
        for k = 1:K
            log_Phi_y =  -0.5*log(2*pi) - 0.5*log(estimated_sigma2(k)) -0.5*((Y - estimated_beta0(k)*ones(N,1) - X*estimated_Beta(:,k)).^2)/estimated_sigma2(k);
            log_Phi_xy(:,k) = log(estimated_weights(:,k)) + log_Phi_y;
        end
        %------------------------------------------------------------------
        log_sum_PhixPhiy = logsumexp(log_Phi_xy,2);
        loglik           = sum(log_sum_PhixPhiy);
        log_Tau = log_Phi_xy - log_sum_PhixPhiy*ones(1,K);
        Tau = exp(log_Tau);
        Tau = Tau./(sum(Tau,2)*ones(1,K));
        pred_labels = MAP(Tau);

        Y_     = estimated_beta0  + X * estimated_Beta;
        Y_pred = sum(Y_ .* estimated_weights,2);

        RPE_   = RPE(Y, Y_pred);
        corr_  = corr(Y ,Y_pred);
        [ARI_, RI_]  = RandIndex(true_labels, pred_labels);
        ClustErr_  = clusteringError(true_labels, pred_labels);
 
        
    end
    
    if verbose
        fprintf('  Estimation evaluation\n');
        fprintf('     Learning Time  : %4.3f (s)\n', learning_time);
        fprintf('     Trans. Distance: %5.3f \n', trandis);
        fprintf('     Log-likelihood : %7.3f \n', loglik);
        fprintf('     MSE with truth : %4.3f \n', mse_param);
        fprintf('  Prediction evaluation\n');
        fprintf('     RelativePredErr: %1.3f \n', RPE_);
        fprintf('     Correlation    : %1.3f \n', corr_);
        fprintf('     Rand Index (RI): %1.3f \n', RI_);
        fprintf('     Adjusted RI    : %1.3f \n', ARI_);
        fprintf('     ClusteringErr  : %1.3f (%%)\n\n', ClustErr_);
    end

    metrics.learning_time = learning_time;
    metrics.trandis = trandis;
    metrics.loglik = loglik;
    metrics.mse_param = mse_param;
    metrics.RPE_ = RPE_;
    metrics.corr_ = corr_;
    metrics.RI_ = RI_;
    metrics.ARI_ = ARI_;
    metrics.ClustErr_ = ClustErr_;

    prediction.Y_pred = Y_pred;
    prediction.pred_labels = pred_labels;
    
    
end