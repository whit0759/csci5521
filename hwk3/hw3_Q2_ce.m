%% CSCI 5521: Homework 3
%
% Homework 3, Question 2, Parts (c),(e)
%
% Christopher White
% April 1, 2019

%% Question 2(c)
image_file = 'goldy.bmp';
flag = 0;
k = 7;

try
    [h,m,Q] = EMG(flag, image_file, k);
catch ME
    warning('EMG failed with flag=0 and k=7');
end

[img, cmap] = imread(image_file);
img_rgb = ind2rgb(img,cmap);
img_double = im2double(img_rgb);

img_vector = reshape(img_double, [], 3);

[idx,m,~,~] = kmeans(img_vector, k, 'MaxIter', 100, 'EmptyAction','singleton');

newvec = m(idx,:);

new_img_dbl = reshape(newvec, size(img_double));
[new_img, new_cmap] = rgb2ind(new_img_dbl,length(cmap));

figure();
imagesc(new_img);
xticklabels([]);
yticklabels([]);
title(sprintf('Reconstructed Image (%s) by Kmeans for k=%d',image_file,k));

%% Question 2(e)
flag = 1;

try
    [h,m,Q] = EMG(flag, image_file, k);
catch ME
    warning('EMG failed with flag=1 and k=7');
end