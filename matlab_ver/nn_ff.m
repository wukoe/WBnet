% Feed-forward network, forward propagation.
%   [X,NN]=nn_ff(NN,X,varargin) varin=I
function [X,NN]=nn_ff(NN,X,flagTrain,varargin)
lAmt=numel(NN);
sAmt=size(X,1);

if ~isempty(varargin)
    activeI=varargin{1};
    bDrop=true;
else
    bDrop=false;
end

% Input layer
NN{1}.act=X;
if bDrop
    for li=2:lAmt
        X = X * NN{li}.W(activeI{li-1},activeI{li});
        if NN{li}.bBias
            X = X + repmat(NN{li}.B(activeI{li}), sAmt,1);
        end
        NN{li}.zin(:,activeI{li})=X;% total input is necessary to record too.
        X = NN{li}.f(X); %* another form arrayfun(@f,X)
        NN{li}.act(:,activeI{li})=X;
    end
else
    for li=2:lAmt
        X = X * NN{li}.W;
        if NN{li}.bBias
            X = X + repmat(NN{li}.B, sAmt,1);
        end
        if flagTrain
            NN{li}.zin=X;% total input is necessary to record too.
            X = NN{li}.f(X); %* another form arrayfun(@f,X)
            NN{li}.act=X;
        else
            X = NN{li}.f(X); %* another form arrayfun(@f,X)
            NN{li}.zin=[];
            NN{li}.act=[];
            NN{li}.err=[];
            NN{li}.vW=[]; NN{li}.momentumW=[];
            NN{li}.vB=[]; NN{li}.momentumB=[];
        end
    end
end

end