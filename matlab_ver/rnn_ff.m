% process one time series sample at a time.
function [Y,nnc]=rnn_ff(nnc,X,flagTrain)
[sAmt,dim]=size(X);

if flagTrain
    nnc.zin=zeros(sAmt,dim);
end
if nnc.bBias
    repB=nnc.B;
else
    repB=0;
end
R=nnc.R; W=nnc.W;
Y=zeros(sAmt,dim);

% Init hidden unit activity.
if isfield(nnc,'act')
    act=nnc.act(end,:);
else
    act=zeros(1,dim);
end
% Prop.
if flagTrain
    for si=1:sAmt
        zin = act*R + X(si,:)*W;
        zin = zin + repB;
        act = nnc.f(zin);
        Y(si,:) = act;
        nnc.zin(si,:) = zin;
    end
else
    for si=1:sAmt
        zin = act*R + X(si,:)*W;
        zin = zin + repB;
        act = nnc.f(zin);
        Y(si,:) = act;
    end
end

if flagTrain
    nnc.act=Y;
end

end