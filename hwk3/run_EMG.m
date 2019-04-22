%% CSCI 5521: Homework 3
%
% Christopher White
% April 1, 2019

%% Question 2(a) and 2(b)
image_file = 'stadium.bmp';
flag = 0;

for k=[4,8,12]
    try
        [h,m,Q] = EMG(flag, image_file,k);
    catch ME
        [h,m,Q] = EMG(flag, image_file,k);
    end
    
end

%% Question 2(c) and 2(e)
run('hw3_Q2_ce.m');
