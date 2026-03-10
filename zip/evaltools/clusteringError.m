function pourcent_misclassified = clusteringError(true_labels, estimated_labels)



    crtb = crosstab(true_labels, estimated_labels);
    K = length(crtb);
    a = perms([1:K]);
    
    try
        correct = 0;
        for i=1:factorial(K)
            cross_sum = trace(crtb(:,a(i,:)));
            if cross_sum > correct
                correct = cross_sum;
            end
        end


        n=length(true_labels);
        misclassified = n-correct;
        pourcent_misclassified = (misclassified/n)*100; 
    catch err
        fprintf('Degenrated number of clusters');
        pourcent_misclassified = 100;
    end

end