function options = get_options(settings)

    if strcmp(settings, 'default')
        options.DME_tol     = 1e-6;
        options.DME_verbose = 1; % {0, 1, 2}
        options.DME_maxiter = 500;
        options.DME_tries   = 1;
        options.tol         = 1e-6;
        options.verbose     = 1; % {0, 1, 2}
        options.maxiter     = 4000;
        options.nb_EM_runs  = 2;
        options.IRLS_threshold = 1e-8;
        options.initialize_strategy = 'LR';
        options.LASSO = false;

    else
        %
    end

end
