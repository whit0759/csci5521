%%CSCI 5521 Homework 4: Problem 2(a)
% Christopher White
% April 22, 2019

%% Training Function
train_txt = 'optdigits_train.txt';
val_txt = 'optdigits_valid.txt';
test_txt = 'optdigits_test.txt';

% Number of ouputs equals number of classes
k=10;

% Sweep number of hidden units
msweep = [3:3:18];

ErrMat = zeros(length(msweep),2);
b=0;
for m=msweep
    b=b+1;
    [z,wtmp,vtmp,errtrain,errval] = mlptrain(train_txt, val_txt,m,k);
    ErrMat(b,1) = errtrain;
    ErrMat(b,2) = errval;
    
    if b>1
        if ErrMat(b,2) < ErrMat(b-1,2)
            w = wtmp;
            v = vtmp;
        end
    else
        w = wtmp;
        v = vtmp;
    end
end

semilogy(msweep, ErrMat);
title('Error Results versus Number of Hidden Units (m)');
xlabel('Hidden Units (m)');
ylabel('Error Rate');
legend('Training Set', 'Validation Set');

%% Testing Function

ztest = mlptest(test_txt, w, v);
