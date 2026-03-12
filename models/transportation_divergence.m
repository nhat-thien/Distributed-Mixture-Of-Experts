function tran_dis = transportation_divergence(large_MoE, reduced_MoE, plan, X_val)

    [n, d] = size(X_val);
    KM  = length(large_MoE.variances);
    K   = length(reduced_MoE.variances);
    tran_dis = [];
    

    for l=1:KM
        
        expert1.xBeta  = large_MoE.experts(:,l);
        expert1.sigma2 = large_MoE.variances(l);
        
        for k=1:K
            
            expert2.xBeta =  reduced_MoE.experts(:,k);
            expert2.sigma2 = reduced_MoE.variances(k);
            
            [~, KLdisvec] = KL_divergence(expert1, expert2, X_val);
            
            if length(plan{l,k}) > 1
                expXA  = exp(X_val*reshape(plan{l,k}, d, K));
                weight = expXA ./ sum(expXA,2);
                weight = weight(:,k);
            else
                weight = 0;
            end
            
            KLdisvec = weight .* KLdisvec;
            tran_dis = [tran_dis sum(KLdisvec)];
            
        end
    end
    
    tran_dis = sum(tran_dis);

end