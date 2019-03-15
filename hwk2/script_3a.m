%% Script to generate the results for problem 3(a).

%% Load datasets
load face_train_data_960.txt;
load face_test_data_960.txt;

%% Create a combined data set
data = [face_train_data_960(:,1:end-1); face_test_data_960(:,1:end-1)];

%% Run PCA on the combined data
[PC, vars] = myPCA(data, 5);

%% Visualize the eigen-faces
figure('Units','inches','Position',[1 1 8 6]);

for n=1:5
    subplot(2,3,n); 
    imagesc(reshape(PC(:,n),32,30)');
    title(sprintf('EigenFace #%d',n));
    daspect([1 1 1]);
end
saveas(gcf,'pca_3a.png');
% close(gcf);