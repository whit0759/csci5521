function [PC1, PC2, mu1, mu2, S1, S2, err] = MultiGaussian(training_data,...
    testing_data, model_number)
%MultiGaussian
%   training_data: data to train the discriminant values on
%   testing_data: data to test the model on
%   model_number: the model to use, i.e. 1,2,3

%% Create Tables for the data
train_table = array2table(training_data);
test_table = array2table(testing_data);
train_table.Properties.VariableNames(end)={'class'};
test_table.Properties.VariableNames(end)={'class'};
N = size(train_table,1);
d = size(train_table,2)-1;

%% Calculate Estimates for Priors and Means
train_c1 = train_table(train_table.class==1, :);
train_c2 = train_table(train_table.class==2, :);
N1 = size(train_c1,1);
N2 = size(train_c2,1);

PC1 = N1/N;
PC2 = N2/N;

mu1 = mean(train_c1{:,1:end-1},1);
mu2 = mean(train_c2{:,1:end-1},1);
mu = mean(train_table{:,1:end-1},1);

%% Train Data based on Class
x = train_table{:,1:d};
x1 = train_c1{:,1:d};
x2 = train_c2{:,1:d};

if model_number==1
    S1 = cov(x1);
    S2 = cov(x2);
elseif model_number==2
    S = cov(x);
    S1 = S;
    S2 = S;
elseif model_number==3
    a1 = var(reshape(x1,1,[]));
    S1 = a1*eye(d);
    a2 = var(reshape(x2,1,[]));
    S2 = a2*eye(d);
end

%% Test Data
xt = test_table{:,1:d};
Nt = size(xt,1);

g1 = zeros(1,N);
g2 = zeros(1,N);
% Model 1 and 3 have the same discriminant
if model_number==1 || model_number==3
    for t=1:N
        g1(t) = log(PC1)-d/2*log(2*pi)-0.5*log(det(S1))-0.5*(xt(t,:)-mu1)*inv(S1)*(xt(t,:)-mu1)';
        g2(t) = log(PC2)-d/2*log(2*pi)-0.5*log(det(S2))-0.5*(xt(t,:)-mu2)*inv(S2)*(xt(t,:)-mu2)';
    end
elseif model_number==2
    for t=1:N
        g1(t) = log(PC1)-0.5*(xt(t,:)-mu1)*inv(S)*(xt(t,:)-mu1)';
        g2(t) = log(PC2)-0.5*(xt(t,:)-mu2)*inv(S)*(xt(t,:)-mu2)';
    end
end

% Identify the class (adding 1 makes the results 1 or 2 instead of 0 or 1)
class = 1+(g2>=g1);

% Compare the test results to correct results and sum the errors.
err = sum(test_table.class(:) ~= class')/Nt;


%% Print Results
fprintf('TABLE OF TEST RESULTS\n\n');
fprintf('P(C1): %0.5g\n',PC1);
fprintf('P(C2): %0.5g\n',PC2);
fprintf('Error Rate: %0.5g\n',err);
fprintf('mu1: %10.5g %10.5g %10.5g %10.5g %10.5g %10.5g %10.5g %10.5g \n',mu1);
fprintf('mu2: %10.5g %10.5g %10.5g %10.5g %10.5g %10.5g %10.5g %10.5g \n',mu2);
fprintf('------------------------------------------\n');
fprintf('S1: %10.5g %10.5g %10.5g %10.5g %10.5g %10.5g %10.5g %10.5g \n',S1);
fprintf('------------------------------------------\n');
fprintf('S2: %10.5g %10.5g %10.5g %10.5g %10.5g %10.5g %10.5g %10.5g \n',S2);



end

