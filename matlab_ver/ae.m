% auto encoder
% method 1: train as usual FF, weight of laye 1 & 2 not required to be
% same.
function [NN,varargout]=ae(NN,NP,opt,X)
sNum=size(X,1);
lNum=3;
if nargout==2 % training error
    flagTE=true;
else
    flagTE=false;
end
% making sure
NN{3}.W=NN{2}.W';

% Proc
batchAmt=floor(sNum/opt.batchSize);
segm=cutseg([1,sNum],opt.batchSize,'equal');

% Training
if flagTE
    trerr=zeros(batchAmt*opt.epochAmt,1);
end
ct=0;
for ei=1:opt.epochAmt
    sampI=randperm(sNum);
    for bi=1:batchAmt
        I=sampI(segm(bi,1):segm(bi,2)); % current batch for stochastic training.  
        ct=ct+1;
        
        % Feed forward
        [bp,NN]=nn_ff(NN,X(I,:));
        if flagTE % get cost
            trerr(ct)=mean(NP.cost(bp,X(I,:)));
        end
        
        % Get cost derivative (error) at output layer
        detcost=NP.detCost(bp,X(I,:));% dC/d(a_L)
        % Error at output layer
        if nargin(NN{lNum}.df)==1
            temp=NN{lNum}.df(NN{lNum}.zin); % derivative of activation function to output layer in-sum.
            NN{lNum}.err = detcost .* temp;
        else % if f'() of output have 2 input
            NN{lNum}.err = NN{lNum}.df(bp,detcost);
        end

        % Error Backprop
        NN=nn_bp(NN);
        
        % Weight-tied update
        li=3;
        NN{li}.vW = NN{li-1}.act' * NN{li}.err;% =de/d(W_k)
        NN{li}.vB = sum(NN{li}.err,1);
        li=2;
        NN{li}.vW = NN{li-1}.act' * NN{li}.err;% =de/d(W_k)
        NN{li}.vB = sum(NN{li}.err,1);
        
        % 1/3 update both layer
        vW = NN{2}.vW + NN{3}.vW';
        NN{2}.W = NN{2}.W - NP.lr(2)*vW;
        NN{3}.W = NN{2}.W';
%         % 2/3 only update by X->H
%         vW = NN{3}.vW; 
%         NN{3}.W = NN{3}.W - 2*NP.lr(3)*vW;
%         NN{2}.W = NN{3}.W';
%         % 3/3 only update by H->R
%         vW = NN{2}.vW; 
%         NN{2}.W = NN{2}.W - 2*NP.lr(2)*vW;
%         NN{3}.W = NN{2}.W';
        % e/3
        
        NN{2}.B = NN{2}.B - NP.lr(2)*NN{2}.vB;
        NN{3}.B = NN{3}.B - NP.lr(3)*NN{3}.vB;     
        
    end
end

% % Proc
% NP=net_lr(NP,opt.lr);
% NN=nn_train(NN,NP,X,X,opt);

if flagTE
    plot(trerr);
    varargout{1}=trerr;
end
end