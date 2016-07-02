% downward propagate to next layer 
% stochastic sampling or expectation
function X = rbm_down(rbm, X, bStoc)
sAmt=size(X,1); % row as unit
vAmt=numel(rbm.vB);
if isempty(bStoc)
    bStoc=true;
end

% Get probability of Hidden units (=1)
ne= repmat(rbm.vB,sAmt,1) + X*rbm.W';
P=1./(1+exp(-ne));
if bStoc
    % use stochastic binary states for hidden unit.
    X=ones(sAmt,vAmt); X(rand(sAmt,vAmt)>P)=0;
else
    X=P;
end

end