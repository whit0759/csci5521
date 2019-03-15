%% Script to generate the results for problem 2(d) and 2(e).

%% Load datasets
load optdigits_train.txt;
load optdigits_test.txt;

% Split the data from the class, which is the last column.
train = optdigits_train(:,1:end-1);
train_class = optdigits_train(:,end);
test = optdigits_test(:,1:end-1);
test_class = optdigits_test(:,end);

%% Find projection matrix
[proj_mat, eigvals] = myLDA(optdigits_train, 9);

%% Calculate projections
Ztest = real(transpose(proj_mat(:,:)'*test'));
Ztrain = real(transpose(proj_mat(:,:)'*train'));

fprintf('RESULTS TABLE FOR 2(D)\n--------------\n');
fprintf('L\tk\tError\n--------------\n');
for L=[2,4,9]
    for k=[1,3,5]
        [~,err]=myKNN([Ztrain(:,1:L), train_class], [Ztest(:,1:L), test_class], k);
        fprintf('%d\t%d\t%g\n',L,k,err);
    end
end

%% Plot the Training Data
figure('Units','inches','Position',[1 1 8 6]);
scatter(Ztrain(:,1), Ztrain(:,2),[],train_class, 'filled');
title({'Linear Discriminant Analysis for Training Data','Number of Components = 2'});
xlabel('Projection 1');
ylabel('Projection 2');
ax = gca;
ax.FontSize=14;
Ntrain=size(Ztrain,1);
for j=1:round(sqrt(Ntrain)):Ntrain
    text(Ztrain(j,1),Ztrain(j,2),num2str(train_class(j)),...
        'FontSize',16,'FontWeight','bold');
end
saveas(gcf,'train_2e.png');
%close(gcf);

%% Plot the Testing Data
figure('Units','inches','Position',[1 1 8 6]);
scatter(Ztest(:,1), Ztest(:,2),[],test_class, 'filled');
title({'Linear Discriminant Analysis for Testing Data','Number of Components = 2'});
xlabel('Projectiont 1');
ylabel('Projection 2');
ax = gca;
ax.FontSize=14;
Ntest=size(Ztest,1);
for j=1:round(sqrt(Ntest)):Ntest
    text(Ztest(j,1),Ztest(j,2),num2str(test_class(j)),...
        'FontSize',16,'FontWeight','bold');
end
saveas(gcf,'test_2e.png');
%close(gcf);