function [z] = mlptest(test_data, w, v)
%MLPTEST Function to test the data to generate a matrix of hidden unit
%values(z)
%

    test_set = load(test_data);

    m = size(w,1); % Hidden Units
    k = size(v,1); % Outputs
    d = size(w,2)-1; % Inputs
    n = size(test_set,1); % Test samples

    inputs = [ones(n,1), test_set(:, 1:d)];
    test_cat = test_set(:,end);

    z=ones(n,m+1);
    err = 0;

    %% Evaulate Test Set
    for t = 1:n
        xt = inputs(t,:);
        rt = zeros(1,k);
        rt(test_cat(t)+1)=1;

        % Calculate z_h with zh(1)=1 always
        for h=2:m+1
            z(t, h) = ReLU(w(h-1,:),xt);
        end

        % Calculate the outputs using the softmax function
        o=v*z(t,:)';
        y=exp(o)/sum(exp(o));

        % Calculate the error for this iteration
        err = err-log(y(rt==1));
    end

    err = err/n;

    fprintf('Error Report          \n');
    fprintf('----------------------\n');
    fprintf('Inputs: %d\n', d);
    fprintf('Outputs: %d\n', k);
    fprintf('Hidden Units: %d\n', m);
    fprintf('Error Rate: %0.0g\n', err);

    %% SUPPLEMENTAL FUNCTIONS
    function out = ReLU(wh, xt)
        % ReLU Activation Function
        x = wh*xt';
        
        if x<0
            out = 0;
        else
            out = x;
        end
    end

end

