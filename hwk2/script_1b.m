%% Script to generate the results for problem 1(b).

%% Load datasets
load training_data1.txt;
load test_data1.txt;
load training_data2.txt;
load test_data2.txt;
load training_data3.txt;
load test_data3.txt;


%% Run KNN on the training and test data
ModelNumber = [1,2,3]';
ErrTest1 = zeros(3,1); 
ErrTest2 = zeros(3,1);
ErrTest3 = zeros(3,1);
for n=1:3
    [PC11,PC21,mu11,mu21,S11,S21,ErrTest1(n)]=MultiGaussian(training_data1, test_data1, n)
    [PC12,PC22,mu12,mu22,S12,S22,ErrTest2(n)]=MultiGaussian(training_data2, test_data2, n)
    [PC13,PC23,mu13,mu23,alpha1,alpha2,ErrTest3(n)]=MultiGaussian(training_data3, test_data3, n)
end

results = table(ModelNumber, ErrTest1, ErrTest2, ErrTest3);
disp(results);

    