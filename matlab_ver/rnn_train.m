% Train RNN
function [nnc,outerr,P]=rnn_train(nnc,X,Y,opt)
sAmt=size(X,1);
hNum=size(nnc.R,1); % number of hidden units.

% Propagate sample in RNN.
[P,nnc]=rnn_ff(nnc,X,true);

% Backprop prepare.
Rt=nnc.R'; % recurrent conn transpose
Wot=eye(hNum); % output conn transpose
dfzin=nnc.df(nnc.zin);
% * lower and higher bound of df
% dfzin=max(dfzin,0.01);
% dfzin=min(dfzin,10);

% Get err from output connection
detcost=P-Y; % cost
outerr = detcost * Wot .* dfzin;
% outerr(1:end-1)=0;

% Do recurrent layer error backprop through time.
nnc.err=zeros(sAmt,hNum);
nnc.err(sAmt,:)=outerr(sAmt,:);
for si=sAmt-1:-1:1
    temp = outerr(si,:) + nnc.err(si+1,:)*Rt;% error from output(t) + from next step(t+1).
    nnc.err(si,:) = temp .* dfzin(si,1); % error at this step.
%     temp = min(temp,5); for restrict the range of error.
%     temp = max(temp,-5);
end

% Update for the parameter.
dR = nnc.act(1:sAmt-1,:)'*nnc.err(2:sAmt,:);
dW = X'*nnc.err;
nnc.R = nnc.R - opt.lr * dR;%/(sAmt-1);
nnc.W = nnc.W - opt.lr * dW;%/sAmt;
if nnc.bBias
    dB = sum(nnc.err,1);
    nnc.B = nnc.B - opt.lr * dB;%/sAmt;
end

end