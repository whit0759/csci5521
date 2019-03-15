%% Script to generate the results for problem 2(b).

%% Load datasets
load optdigits_train.txt;
load optdigits_test.txt;


%% Run KNN on the training and test data
fprintf('ERROR RESULTS FOR 2(A)\n------------------\n');

for k=1:2:7
    [class, err] = myKNN(optdigits_train, optdigits_test,k);
    fprintf('The error rate for k=%d is: %0.5g\n', k, err);
end

    