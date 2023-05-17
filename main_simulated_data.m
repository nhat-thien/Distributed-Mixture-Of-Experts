clear;
close all;
addpath ./
addpath ./data/
addpath ./models/
addpath ./results/
addpath ./stattools/
addpath ./evaltools/
rng('default');
rng(0);

%% Load data --------------------------------------------------------------
load SimulatedData
[n,d] = size(X);
N  = floor(0.9*n); %train size
X_train = X(1:N,:);
Y_train = Y(1:N);
X_test  = X(N+1:n,:);
Y_test  = Y(N+1:n);

%% FIT --------------------------------------------------------------------

model  = 'Distributed_MixtureOfExperts';
model  = 'Global_MixtureOfExperts';
options = get_options('default');
K = 5;
M = 4;
switch model
    case('Distributed_MixtureOfExperts')
        fit    = Distributed_MixtureOfExperts_Gaussian(X_train, Y_train, K, M, options);
    case('Global_MixtureOfExperts')
        fit    = Global_MixtureOfExperts(X_train, Y_train, K, options);
    otherwise
        error('Specify a model')
end

%% PRINT ------------------------------------------------------------------
fprintf('On training data\n') 
[~, ~] = compute_metrics(fit, true_mixture, X_train, Y_train, true_labels(1:N), 1);
fprintf('On testing data \n') 
[~, prediction] = compute_metrics(fit, true_mixture, X_test, Y_test, true_labels(N+1:n), 1);

%% PLOT -------------------------------------------------------------------
close all
figure('Position', [0,100,500,300])
timepoint = randi([1,d]);
gscatter(X_test(:,timepoint),Y(N+1:end), true_labels(N+1:end));
title('True responses in true labels');
ylabel('True Y')
xlabel(['X(:,',num2str(timepoint),')'])

figure('Position', [600,100,500,300])
gscatter(X_test(:,timepoint),prediction.Y_pred, prediction.pred_labels);
title('Predicted responses in predicted labels');
ylabel('Predicted Y')
xlabel(['X(:,',num2str(timepoint),')'])

    





