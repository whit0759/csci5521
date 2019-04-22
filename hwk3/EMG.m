function [ h, m, Q ] = EMG( flag, image_file, k )
%EMG implements the EM algorithm
%   flag:   0: implements the standard EM algorithm
%           1: implements the improved EM algorithm 
%   image: file path to an image
%   k: scalar value of the number of clusters

[img, cmap] = imread(image_file);
img_rgb = ind2rgb(img,cmap);
img_double = im2double(img_rgb);

img_vector = reshape(img_double, [], 3);

[idx,mu_i,~,~] = kmeans(img_vector, k, 'MaxIter', 3, 'EmptyAction','singleton');

% tmp = eye(k);
% tmp2 = tmp(idx,:);
% 
% image_reconst(tmp2, mu_i);
% title(sprintf('Reconstructed Image by Kmeans for k=%d',k));

%%  Initialize Arrays
% 
N = length(idx);
Ni = zeros(k,1);
Si = zeros(3,3,k);
pdf_i = zeros(N,k);
pg_i = zeros(N,k);
ht_i = zeros(N,k);


%% Initialize pi, S, and ht
for i=1:k
    Ni(i) = sum(idx==i);
    xi = img_vector(idx==i,:);
    %Si(:,:,i) = cov(img_vector(idx==i,:));
    Si(:,:,i) = (xi-mu_i(i))'*(xi-mu_i(i))/Ni(i);
end

pi_i = Ni/N;

MAX_ITERS=100;
TOL=1E-3;
Qe=[];
Qm=[];
l=0;

for i=1:k
    pdf_i(:,i) = mvnpdf(img_vector,mu_i(i,:),Si(:,:,i)); 
    pg_i(:,i) = pi_i(i)*pdf_i(:,i);
end

while l<MAX_ITERS
    l = l+1;
   
    %% Expectation Step    
    ht_i = pg_i./sum(pg_i,2);
    
    Ql = 0;
    for t=1:N
        Ql = Ql + ht_i(t,:)*log(pg_i(t,:)');
    end
    
    Qe = [Qe, Ql];
    
    %% Maximization Step
    Ni = sum(ht_i);
    pi_i = Ni/N;
    mu_i = ht_i'*img_vector./sum(ht_i)';

    if flag==0
        for j=1:k
            tmp = img_vector - mu_i(j,:);
            Si(:,:,j) = ((ht_i(:,j).*tmp)'*tmp)/Ni(j);
        end
    elseif flag==1
        lambda = 1e-3;
        for j=1:k
            tmp = img_vector - mu_i(j,:);
            Si(:,:,j) = ((ht_i(:,j).*tmp)'*tmp)/Ni(j) + lambda*eye(3);
        end
    else
        error('Flag must be 0 or 1');
    end
    
    for i=1:k
        pdf_i(:,i) = mvnpdf(img_vector,mu_i(i,:),Si(:,:,i)); 
        pg_i(:,i) = pi_i(i)*pdf_i(:,i);
    end
    
    Ql = 0;
    for t=1:N
        Ql = Ql + ht_i(t,:)*log(pg_i(t,:)');
    end
    
    Qm = [Qm, Ql];
    
    if abs((Qm(end)-Qe(end))/Qe(end))<TOL
        l = MAX_ITERS+1;
    end
end

h = ht_i;

m = mu_i;

Q = Qm;

image_reconst(h,m);
title(sprintf('Reconstructed Image (%s) by Gaussian Mixtures for k=%d', image_file, k));

% plot_log_like();
plot_log_like_E_M();

    function image_reconst(h, m)
        [~,cls] = max(h,[],2);
        newvec = m(cls,:);
        
        new_img_dbl = reshape(newvec, size(img_double));
        [new_img, new_cmap] = rgb2ind(new_img_dbl,length(cmap));
        
        figure();
        imagesc(new_img);
        xticklabels([]);
        yticklabels([]);
    end
    
    function plot_log_like()
        figure();
        plot(Q);
        title(sprintf('Complete Log-Likelihood for k=%d', k));
        xlabel('Iteration');
        ylabel('Q(\Phi|\Phi^l)');
    end
    
    function plot_log_like_E_M()
        figure();
        plot(Qe);
        hold on;
        plot(Qm);
        legend('E-Step','M-Step');
        title(sprintf('Complete Log-Likelihood for k=%d with E-Step and M-Step', k));
        xlabel('Iteration');
        ylabel('Q(\Phi|\Phi^l)');
    end

end