%% Script to generate the results for problem 3(b).

%% Load datasets
load face_train_data_960.txt;
load face_test_data_960.txt;

% Split the data from the class, which is the last column.
train = face_train_data_960(:,1:end-1);
train_class = face_train_data_960(:,end);
test = face_test_data_960(:,1:end-1);
test_class = face_test_data_960(:,end);

%% Run PCA on the training data
[PC, vars] = myPCA(train, 0);

% The sum of the vars will normalize the cumsum of them.
cvars = cumsum(vars)/sum(vars);

% Find the index that is above 90%
K = find(cvars>=0.9, 1);

%% Plot the Proportion of Variance
figure('Units','inches','Position',[1 1 8 6]);
plot(cvars);
title({'Proportion of Variance for FACE\_TRAIN\_DATA',sprintf('Optimal K=%d',K)});
xlabel('Eigenvectors');
ylabel('Proportion of Variance');
ax = gca;
ax.FontSize=14;
saveas(gcf,'ftrain_3b.png');
%close(gcf);

%% Project the Test data to the K components
Ztrain = transpose(PC(:,1:K)'*train');
Ztest = transpose(PC(:,1:K)'*test');

fprintf('ERROR RESULTS FOR 3(B)\n------------------\n');
for k=1:2:7
    [~,err]=myKNN([Ztrain, train_class], [Ztest, test_class], k);
    fprintf('Error rate for k=%d: %g\n', k, err);
end
    