% adjust network learning rate
%   NP=net_lr(NP,lr)
function NP=net_lr(NP,lr)
lNum=length(NP.layerSize);
if numel(lr)==1
    for k=2:lNum
        NP.lr(k)=lr;
    end
else
    if numel(lr)~=lNum
        error('lr vector length mismatch');
    end
    for k=2:lNum
        NP.lr(k)=lr(k);
    end
end
end