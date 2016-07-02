% Upward propagate to next layer 
%   X = rbm_up(rbm, X, bStochastic)
% bStochastic controls whether use stochastic sampling or expectation.
function X = rbm_up(rbm, X, bStoc)
sAmt=size(X,1); % row as unit
hAmt=numel(rbm.hB);
if isempty(bStoc)
    bStoc=true;
end

% Get probability of Hidden units =1
ne= repmat(rbm.hB,sAmt,1) + X*rbm.W;
P=1./(1+exp(-ne));
if bStoc
    % use stochastic binary states for hidden unit.
    X=ones(sAmt,hAmt); X(rand(sAmt,hAmt)>P)=0;
else
    X=P;
end

end