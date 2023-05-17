function [plan, distance_matrix] = argmin_transportation_plan(large_mixture, reduced_mixture, X_val)

    L  = length(large_mixture.variances);
    K  = length(reduced_mixture.variances);
    [S,d] = size(X_val);
    M = L/K;
    
    if size(large_mixture.gates,1) == d
        large_mixture.gates = repmat(large_mixture.gates(:),1,K);
    end

    %Compute PI_hat for each x in X_S
    PI_hat  = [];
    for m=1:M
        X_val_times_Gate = X_val*reshape(large_mixture.gates(:,m*K),d,K);
        maxx             = max(X_val_times_Gate, [], 2);
        X_val_times_Gate = X_val_times_Gate - maxx;
        exp_X_val_times_Gate = exp(X_val_times_Gate);
        PI_temp = exp_X_val_times_Gate ./ sum(exp_X_val_times_Gate,2);
        PI_hat = [PI_hat PI_temp];
    end
    %Make sure sum(PI_hat,2) = [1,...,1]'
    PI_hat = PI_hat.*large_mixture.weights;
    PI_hat = PI_hat';
    
    %Compute the tensor of distances [L-K-S]
    distance_matrix = zeros(L,K,S);
    for l=1:L
        
        expert1.xBeta  = large_mixture.experts(:,l);
        expert1.sigma2 = large_mixture.variances(l);
        
        for k=1:K
            
            expert2.xBeta =  reduced_mixture.experts(:,k);
            expert2.sigma2 = reduced_mixture.variances(k);

            [~, KLdisvec] = KL_distance(expert1, expert2, X_val);
            distance_matrix(l,k,:) = KLdisvec;
        end
        
    end
    
    %Assign plan for each x in X_S
    plan  = zeros(L,K,S);
    
    [~,mindis_index] = min(distance_matrix,[],2);
    mindis_index = squeeze(mindis_index);
    for l=1:L
        for k=1:K
            plan(l,k,:) = PI_hat(l,:).*(mindis_index(l,:)==k);
        end
    end
    
end



