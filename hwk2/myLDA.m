function [ proj_mat, eigvals ] = myLDA(data, num_principal_components)
%MYLDAA Calculates the linear discriminant based on the data
%   The data is training data. 

%% Create a table from the data
%   It is easier to select by class if the data is tabularized.
dtable = array2table(data);
dtable.Properties.VariableNames(:,end)={'class'};

K = max(dtable.class);
[t,d] = size(data(:,1:end-1));

%% Create within class scatter matrix
Sw = zeros(d,d);
mi = zeros(K,d);
Ni = zeros(K,1);

for i=1:K
    cdata = dtable{dtable.class==i, 1:end-1};
    Ni(i) = size(cdata,1);
    mi(i,:) = mean(cdata,1);
    xtm =  cdata-mi(i,:);
    Sw = Sw + (xtm'*xtm);
end

%% Create between class scatter matrix
m=sum(mi,1)/K;

Sb = zeros(d,d);

for j=1:K
    mim = mi(j)-m;
    Sb = Sb + Ni(j)*(mim'*mim);
end

%% Find the specified number of projections
[V,D] = eig(pinv(Sw)*Sb,'vector');

% Sort the eigenvalues
[Dsrt,Ind]  = sort(D,1,'descend');
Vsrt = V(:,Ind);

%   If num_principal_components is greater than zero, return the number
%   specified. Otherwise return all of them.
if num_principal_components>0
    proj_mat = Vsrt(:,1:num_principal_components);
    eigvals = Dsrt(1:num_principal_components);
else
    proj_mat = Vsrt;
    eigvals = Dsrt;
end

