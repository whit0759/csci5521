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

figure;
map = [0 0 0
        1 0 0];
colormap(map);
scatter(X(:,1),X(:,2),[],y,'filled');
xplt = xlim;
yset = ylim;
yplt = -w(1)*xplt/w(2);
line(xplt,yplt);
ylim(yset);
title('Initial Scatter Plot and Weight');
    


while err>0
    errcnt = 0;
    for ii=1:N
        
        wTXii = dot(w,X(ii,:));

        if wTXii*y(ii) <=0
            w = w + y(ii)*X(ii,:)';
        end
        
        if sign(wTXii)+sign(y(ii)) == 0
            errcnt = errcnt+1;
        end
    end
    
    step = step + 1;
    err = errcnt;
    
    if step>50
        break
    end

end

figure;
colormap(map);
scatter(X(:,1),X(:,2),[],y,'filled');
xplt = xlim;
yset = ylim;
yplt = -w(1)*xplt/w(2);
line(xplt,yplt);
ylim(yset);
title('Final Scatter Plot and Weight');

end

