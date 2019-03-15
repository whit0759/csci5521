function [ p1, p2, pc1, pc2 ] = Bayes_learning( training_data, validation_data )
%BAYES_LEARNING Learns probabilites for training data
%   Description: takes in two data files. The training data is used to
%   calculate the probability values and to select the best value for
%   sigma. The validation data is used to assess the quality of the learned
%   parameters.
%
%   Inputs:
%       training_data : data to train the learning algorithm on
%       validation_data : data to validate the learned parameters against
%
%   Outputs:
%       p1 : Learned probabilities for class 1
%       p2 : Learned probabilities for class 2
%       pc1 : Best prior for class 1
%       pc2 : Best prior for class 2

%% Load data into tables
tdata = array2table(training_data);
tdata.Properties.VariableNames(end) = {'class'};

vdata = array2table(validation_data);
vdata.Properties.VariableNames(end) = {'class'};

% Separate the data into Class 1 and Class 2
tc1 = tdata(tdata.class==1,:); % Select data for class 1
tc2 = tdata(tdata.class==2,:); % Select data for class 2

%% Calculate probability estimates
% Count up the ones in each column and divide by the number of columns for
% each class.
Nvars = size(tdata,2)-1;
Ntrain = size(tdata,1);
countc1 = sum(tc1{:,1:Nvars});
countc2 = sum(tc2{:,1:Nvars});

Nc1 = size(tc1,1);
Nc2 = size(tc2,1);

p1c1 = countc1/Nc1;
p1c2 = countc2/Nc2;

sigma = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2, 3, 4, 5, 6];
errTrain = zeros(length(sigma),2);

% Sweep through values of sigma
for j = 1:length(sigma)
    g1 = zeros(Ntrain,1);
    g2 = zeros(Ntrain,1);
    class = ones(Ntrain,1);

    % Calculate the priors from the value of sigma
    c1 = 1-exp(-1*sigma(j));
    c2 = 1-c1;

    % Calculate the P(Ci|x) conditional probabilities for each class's
    % discrimnant
    for n=1:Ntrain
        row = tdata{n,1:Nvars};
        g1(n) = sum(log(p1c1).*row + log(1-p1c1).*(1-row))+log(c1);
        g2(n) = sum(log(p1c2).*row + log(1-p1c2).*(1-row))+log(c2);
        
    end

    % Calculate the selected classes. The +1 makes the entries either 1 or
    % 2 instead of 1 or 0.
    class = class+(g1<g2);

    err1 = tdata.class(:) < class;
    err2 = tdata.class(:) > class;
  
    % Record the error rates for each class at each sigma
    errTrain(j,1) = sum(err1)/Ntrain;
    errTrain(j,2) = sum(err2)/Ntrain;
end

err = sum(errTrain,2);

% Find the index of the sigma minimum
sigma1 = find(err==min(err));

pc1 = 1-exp(-1*sigma(sigma1));
pc2 = exp(-1*sigma(sigma1));

p1 = p1c1;
p2 = p1c2;

%% Validation
Nval = size(vdata,1);

fprintf('TABLE OF ERROR RATES\n\n');
fprintf('P(C1|sigma)\tError Class 1\tError Class 2\n');
fprintf('------------------------------------------\n');

% Sweep sigma
for j=1:length(sigma)
    v1 = zeros(Nval,1);
    v2 = zeros(Nval,1);
    vclass = ones(Nval,1);
    
    c1 = 1-exp(-1*sigma(j));
    c2 = 1-c1;

    % Sweep through validation data and calculate the discriminants
    for n=1:Nval
        row = vdata{n,1:Nvars};
        v1(n) = sum(log(p1c1).*row + log(1-p1c1).*(1-row))+log(c1);
        v2(n) = sum(log(p1c2).*row + log(1-p1c2).*(1-row))+log(c2);

    end
    
    % Determine the class based on the outcome of the discriminants.
    vclass = vclass+(v1<v2);
    
    verr1 = vdata.class(:) < vclass; % Class is 2 should be 1
    verr2 = vdata.class(:) > vclass; % Class is 1 should be 2
    
    fprintf('%0.5g \t %0.5g \t %0.5g\n',c1, sum(verr1)/Nval, sum(verr2)/Nval);

end

end