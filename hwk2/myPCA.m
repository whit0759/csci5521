function [ principal_components, eigvals ] = myPCA(data, num_principal_components)
%MYPCA Calculates the principal components based on the data
%   The data is training data. The principal components are the
%   eigenvectors and eigenvals are sorted.

%% Compute the Covariance matrix
m = mean(data,1);
S = cov(data-m);

%% Compute the Eigenvectors and values
[V,D] = eig(S,'vector');

% Sort the eigenvalues
[Dsrt,Ind]  = sort(D,1,'descend');
Vsrt = V(:,Ind);

%% Return the components and values
%   If num_principal_components is greater than zero, return the number
%   specified. Otherwise return all of them.
if num_principal_components>0
    principal_components = Vsrt(:,1:num_principal_components);
    eigvals = Dsrt(1:num_principal_components);
else
    principal_components = Vsrt;
    eigvals = Dsrt;
end

end