%% Script to generate the results for problem 2(b).

%% Load datasets
load optdigits_train.txt;
load optdigits_test.txt;

% Split the data from the class, which is the last column.
train = optdigits_train(:,1:end-1);
train_class = optdigits_train(:,end);
test = optdigits_test(:,1:end-1);
test_class = optdigits_test(:,end);

%% Run PCA on the training data
[PC, vars] = myPCA(train, 0);

% The sum of the vars will normalize the cumsum of them.
cvars = cumsum(vars)/sum(vars);

% Find the index that is above 90%
K = find(cvars>=0.9, 1);

%% Plot the Proportion of Variance
figure('Units','inches','Position',[1 1 8 6]);
plot(cvars);
title({'Proportion of Variance for OPTDIGITS\_TRAIN','Optimal K=21'});
xlabel('Eigenvectors');
ylabel('Proportion of Variance');
ax = gca;
ax.FontSize=14;
saveas(gcf,'pov_2b.png');
%close(gcf);

%% Project the Test data to the K components
Ztrain = transpose(PC(:,1:K)'*train');
Ztest = transpose(PC(:,1:K)'*test');

fprintf('ERROR RESULTS FOR 2(B)\n------------------\n');
for k=1:2:7
    [~,err]=myKNN([Ztrain, train_class], [Ztest, test_class], k);
    fprintf('Error rate for k=%d: %g\n', k, err);
end
    