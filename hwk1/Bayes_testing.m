function Bayes_testing( test_data, p1, p2, pc1, pc2 )
%BAYES_TESTING Test of learned Bernoulli parameters

tdata = array2table(test_data);
tdata.Properties.VariableNames(end) = {'class'};

Nvars = size(tdata,2)-1;
Ntest = size(tdata,1);

g1 = zeros(Ntest,1);
g2 = zeros(Ntest,1);
class = ones(Ntest,1);

for n=1:Ntest
    row = tdata{n,1:Nvars};
    g1(n) = sum(log(p1).*row + log(1-p1).*(1-row))+log(pc1);
    g2(n) = sum(log(p2).*row + log(1-p2).*(1-row))+log(pc2);

end

class = class+(g1<g2);

err1 = tdata.class(:) < class;
err2 = tdata.class(:) > class;

fprintf('TABLE OF TEST RESULTS\n\n');
fprintf('P(C1|sigma)\tError Class 1\tError Class 2\n');
fprintf('------------------------------------------\n');
fprintf('%0.5g \t %0.5g \t %0.5g\n',pc1, sum(err1)/Ntest, sum(err2)/Ntest);

end