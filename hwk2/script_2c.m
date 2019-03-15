%% Script to generate the results for problem 2(c).

%% Load datasets
load optdigits_train.txt;
load optdigits_test.txt;

% Split the data from the class, which is the last column.
train = optdigits_train(:,1:end-1);
train_class = optdigits_train(:,end);
test = optdigits_test(:,1:end-1);
test_class = optdigits_test(:,end);

%% Run PCA on the training data
[PC, vars] = myPCA(train, 2);

%% Project the Test data to the K components
Ztrain = transpose(PC(:,:)'*train');
Ztest = transpose(PC(:,:)'*test');

%% Plot the Training Data
figure('Units','inches','Position',[1 1 8 6]);
scatter(Ztrain(:,1), Ztrain(:,2),[],train_class, 'filled');
title({'Principal Components for Training Data','Number of Components = 2'});
xlabel('Principal Component 1');
ylabel('Principal Component 2');
ax = gca;
ax.FontSize=14;
Ntrain=size(Ztrain,1);
for j=1:round(sqrt(Ntrain)):Ntrain
    text(Ztrain(j,1),Ztrain(j,2),num2str(train_class(j)),...
        'FontSize',16,'FontWeight','bold');
end
saveas(gcf,'train_2c.png');
%close(gcf);

%% Plot the Testing Data
figure('Units','inches','Position',[1 1 8 6]);
scatter(Ztest(:,1), Ztest(:,2),[],test_class, 'filled');
title({'Principal Components for Testing Data','Number of Components = 2'});
xlabel('Principal Component 1');
ylabel('Principal Component 2');
ax = gca;
ax.FontSize=14;
Ntest=size(Ztest,1);
for j=1:round(sqrt(Ntest)):Ntest
    text(Ztest(j,1),Ztest(j,2),num2str(test_class(j)),...
        'FontSize',16,'FontWeight','bold');
end
saveas(gcf,'test_2c.png');
%close(gcf);