function [z, w, v, varargout] = mlptrain(train_file, val_file, m, k)
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

    % Establish the r^t matrix where each row is a t and a 1 is placed in
    % the corresponding category label, i.e. rt(t, 3)=1 for train(t,65)==3 
    rt_train = zeros(Ntrain,k);
    for l=1:Ntrain
        rt_train(l,train_cat(l)+1) = 1;
    end

    % Create the Adjunct Data arrays
    train_data = [ones(Ntrain,1), train_read(:,1:64)];
    val_data = [ones(Nval,1), val_read(:,1:64)];
        
    Nepoch = 100;
      
    %% TRAIN SEQUENCE
    % Initialize the v_{i,h} and w_{h,j} matrices
    vih = init_mat([k, m+1], -0.01, 0.01);
    whj = init_mat([m, 65], -0.01, 0.01);

    % Initialize the z_h, y, and dv_{i,h}, dw_{h,j} matrices
    zh = ones([1,m+1]);
    y = zeros(k,1);
    dvih = zeros(k,m+1);
    dwhj = zeros(m,65);

    eta = 0.001;
    alpha = 0.5;

    ErrTrain = zeros(1,Nepoch);
    ErrVal = zeros(1,Nepoch);

    for ep=1:Nepoch
        z = ones(Ntrain+Nval, m+1);
        zcnt = 0;

        % Iterate over the training data
        for t=randperm(Ntrain) % Randomly order the samples
            zcnt = zcnt+1;
            xt = train_data(t,:);
            rt = rt_train(t,:);

            % Calculate z_h with zh(1)=1 always
            for h=2:m+1
                zh(h) = ReLU(whj(h-1,:),xt);
            end

            % Calculate the outputs using the softmax function
            o=vih*zh';
            y=exp(o)/sum(exp(o));

            % Calculate dvih for each output
            for i=1:k
                dvih(i,:) = eta*(rt(i) - y(i))*zh;
            end

            % Calculate dwhj for each hidden unit
            for h=2:m+1
                if whj(h-1,:)*train_data(t,:)' <0
                    dwhj(h-1,:) = zeros(size(dwhj(h-1,:)));
                else
                    % Adjust dwhj with Momentum for convergence
                    dwhj(h-1,:) = eta*((rt - y')*vih(:,h))*xt + alpha*dwhj(h-1,:);
                end
            end

            % Update vih and whj
            vih = vih + dvih;
            whj = whj + dwhj;

            % Calculate the error for this iteration
            ErrTrain(ep) = ErrTrain(ep)-log(y(rt==1));
            z(zcnt,:) = zh;
        end  

        %% VALIDATION SEQUENCE
        % Iterate over samples
        for t=randperm(Nval)
            zcnt = zcnt+1;
            vxt = val_data(t,:);
            vrt = zeros(1,k);
            vrt(val_cat(t)+1)=1;

            % Calculate z_h with zh(1)=1 always
            for h=2:m+1
                zh(h) = ReLU(whj(h-1,:),vxt);
            end

            % Calculate the outputs using the softmax function
            o=vih*zh';
            vy=exp(o)/sum(exp(o));

            % Calculate the error for this iteration
            ErrVal(ep) = ErrVal(ep)-log(vy(vrt==1));
            z(zcnt,:) = zh;
        end
    end

    ErrTrain = ErrTrain(end)/Ntrain;
    ErrVal = ErrVal(end)/Nval;

    if nargout==5
        varargout{1} = ErrTrain;
        varargout{2} = ErrVal;
    end
    
    fprintf('Hidden Units | Training Error | Validation Error\n');
    fprintf('------------------------------------------------\n');
    fprintf('%10.0d | %14.5g | %14.5g\n', m, ErrTrain, ErrVal);
        
    w = whj;
    v = vih;
    
    %% SUPPLEMENTAL FUNCTIONS
    function mat = init_mat(size, minval, maxval)
        % Function to initialize a matrix
        mat = minval + (maxval-minval).*rand(size);
    end

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

