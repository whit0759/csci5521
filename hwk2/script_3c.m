%% Script to generate the results for problem 3(c).

%% Load datasets
load face_train_data_960.txt;
load face_test_data_960.txt;

% Split the data from the class, which is the last column.
train = face_train_data_960(:,1:end-1);
train_class = face_train_data_960(:,end);
test = face_test_data_960(:,1:end-1);
test_class = face_test_data_960(:,end);

data = [train; test];

%% Run PCA on the training data
[PC, vars] = myPCA(data, 100);

%% Create back-projected data 

% Calculate mean of data set
m = mean([train;test],1);

for K=[10, 50, 100]
    % Project 5 rows of data
    zt = transpose(PC(:,1:K)'*(data(1:5,:)-m)');
    
    xhat = transpose(PC(:,1:K)*zt');
    
    figure(K);
    
    for n=1:5
        subplot(2,5,n); 
        imagesc(reshape(data(n,:),32,30)');
        title({'Original',sprintf('Face #%d',n)});
        daspect([1 1 1]);
        
        subplot(2,5,n+5); 
        imagesc(reshape(xhat(n,:),32,30)');
        title({'Reconstructed',sprintf('Face #%d',n)});
        daspect([1 1 1]);
    end
    
    saveas(gcf,sprintf('back_face_%d.png',K));
end