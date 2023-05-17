function [beta0_out, Beta_out, alpha0_out, Alpha_out, sigma2_out] = identify_network(beta0, Beta, alpha0, Alpha, sigma2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Make Expert and Gating network to be identified by re-ordering according
% to beta0's and/or sigma2's (for Gaussian experts)
% Ref: W.Jiang, M.Tanner, On the Identifiability of Mixtures-of-Experts
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    q = size(Alpha,1);

    % Permutate the sigma2 and get the order
    if length(beta0) == length(unique(beta0))
        [beta0_out, order] = sort(beta0);
        sigma2_out = sigma2(order);
    else %Permutate the beta0 and get the order
        [sigma2_out, order] = sort(sigma2);
        beta0_out = beta0(order);
    end
    Beta_out = Beta(:,order);

    % Pad the Kth gate with zeros
    alpha0_out = [alpha0 0];
    alpha0_out = alpha0_out(order);
    alpha0_out = alpha0_out - alpha0_out(end);

    % Initialize the gates
    Alpha_out = [Alpha zeros(q,1)];
    Alpha_out = Alpha_out(:,order);
    Alpha_out = Alpha_out - Alpha_out(:,end);

    % Remove the Kth gate
    alpha0_out = alpha0_out(1:end-1);
    Alpha_out = Alpha_out(:,1:end-1);


end