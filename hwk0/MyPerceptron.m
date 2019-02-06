function [w, step]= MyPerceptron(X, y, w0)
%MYPERCEPTRON Takes in data and creates a weight vector
%   Author: Christopher White
%   Date: February 1, 2019
%
% Inputs:
%   X: feature matrix
%   y: label vector
%   w0: initial weights
%
% Outputs:
%   w: weight vector
%   step: number of steps to converge

w = w0;
step = 0;
err = 1;

N = length(y);

while err>0
    for ii=1:N,

        wTXii = dot(w,X(ii,:));

        if wTXii*y(ii) <=0,
            w = w + y(ii)*X(ii,:);
        end
    end

end

end

