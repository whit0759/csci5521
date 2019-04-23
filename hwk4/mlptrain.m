function [z, w, v] = mlptrain(train_file, val_file, m, k)
%MLPTRAIN Function to train values to input data
%   The Problem 2 training of the Multilayer Perceptron is implemented in
%   this function. The inputs are the data, the number of hidden units (m),
%   and the number of output units (k). The error rates are printed for the
%   validation data.

train_read = load(train_file);
val_read = load(val_file);

train_cat = train_read(:,65);
Ntrain = length(train_cat);
val_cat = val_read(:,65);
Nval = length(val_cat);

rt_train = zeros(Ntrain,k);
for l=1:Ntrain
    rt_train(l,train_cat(l)+1) = 1;
end

train_data = [ones(Ntrain,1), train_read(:,1:64)];
val_data = [ones(Nval,1), val_read(:,1:64)];

vih = init_mat([k, m+1], -0.01, 0.01);
whj = init_mat([m, 65], -0.01, 0.01);

zh = ones([1,m+1]);
y = zeros(k,1);
dvih = zeros(k,m+1);

eta = 1E-5;

Err = 0;

while del < tol
    for t=1:Ntrain
        for h=1:m
            zh(h) = ReLU(whj(h,:),train_data(randi(Ntrain),:));
        end
        
        for i=1:k
            y(i) = vih(i,:)*zh';
            dvih(i,:) = eta*(rt_train(t,i) - y(i))*zh':
        end
        
        for h=1:m
            dwhj(h,:) = eta*((rt_train(t,i) - y(i))*vih(:,h))*(zh(h).*(1-zh(h)))*train_data(t,:)';
        end
        
        for i=1:k
            vih(i,:) = vih(i,:)+dvih(i,:);
        end
        
        for h=1:m
            whj(h,:) = whj(h,:) + dwhj(h,:);
        end
        
        Err = Err + rt_train(t,:)*log(y');    
    end
    
    
    
end

    function mat = init_mat(size, minval, maxval)
        % Function to initialize a matrix
        mat = minval + (maxval-minval).*rand(size);
    end

    function zh = ReLU(wh, xt)
        % ReLU Activation Function
        x = wh*xt';
        
        if x<0
            zh = 0;
        else
            zh = x;
        end
    end

end

