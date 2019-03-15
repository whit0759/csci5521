function [class, err] = myKNN(training_data, test_data, k)
%myKNN Implementation of k-nearest neighbor
%   training_data: data to train on
%   test_data: test data
%   k: k-value to use

%% Calculate dimensions and class array
[Ntrain,dtrain] = size(training_data(:,1:end-1));
train_class = training_data(:,end);
[Ntest, dtest] = size(test_data(:,1:dtrain));
test_class = test_data(:,end);

%% Create distance array
class = zeros(Ntest,1);
dist = zeros(Ntrain, Ntest);

% Find the Euclidean distance using norm() from each training data point
% to each test data point
for i=1:Ntest
    for j=1:Ntrain
        dist(j,i) = norm(training_data(j,1:end-1)-test_data(i,1:end-1));
    end
end

%% Find the minimum indicies
[min_val, min_ind] = mink(dist,k,1);

for m=1:Ntest
    neighbors=zeros(k,1);
    for n=1:k
        neighbors(n)=train_class(min_ind(n,m));
    end
    
    % The class is the mode of the neighbors' classes.
    class(m) = mode(neighbors);
end

err = sum(class~=test_class)/Ntest;

end

