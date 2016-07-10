% process one time series sample at a time.
function [Y,nnc]=rnn_ff(nnc,X,flagTrain)
sAmt=size(X,1);
hNum=size(nnc.W,2);

if flagTrain
    nnc.zin=zeros(sAmt,hNum);
end
if nnc.bBias
    repB=nnc.B;
else
    repB=0;
end
R=nnc.R; W=nnc.W;
Y=zeros(sAmt,hNum);

% Init hidden unit activity.
if isfield(nnc,'act')
    act=nnc.act(end,:);
else
    act=zeros(1,hNum);
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
% Update activity state.
if flagTrain
    nnc.act=Y;
else
    nnc.act=act;
end

end