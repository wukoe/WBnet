% train DBN
%   [NN]=dbn_train(NN,NP,X,method,opt)
% method = {'sae','srbm'}
function [NN]=dbn_train(NN,NP,X,method,opt)
lAmt=numel(NN);
sNum=size(X,1);

% % Init network
% opt=struct();
opt.unitFunc='tanh';
opt.unitFunc_output='tanh';
opt.costFunc='mse';
opt.dropr=0;
opt.momentum=0;
opt.regular='';
opt.batchSize=10;
opt.epochAmt=ceil(5000/sNum);

if strcmp(method,'sae')
    for li=2:lAmt
        opt.lr='auto';
        [nn,np,opt]=net_setup([NP.layerSize(li-1),NP.layerSize(li),NP.layerSize(li-1)],opt);
        % change reconstruction layer li-1's bias and weight
        nn{2}.W=NN{li}.W;
        nn{3}.W=NN{li}.W';
        nn{2}.B=NN{li}.B;
        if isfield(NN{li-1},'rB')
            nn{3}.B=NN{li-1}.rB;
        else
            nn{3}.B=zeros(1,NP.layerSize(li-1)); %for decoding layer
        end
        
        % Training
        nn=ae(nn,np,opt,X);
        NN{li}.W=nn{2}.W;
        NN{li}.B=nn{2}.B;
        NN{li-1}.rB=nn{3}.B;
        
        % Propagate samples
        X=nn_ff(nn(1:2),X);
    end
    
    %%% build block: RBM method
elseif strcmp(method,'srbm')
    % Init 1st layer
    if ~isfield(NN{1},'B')
        NN{1}.B=zeros(1,NP.layerSize(1));
    end
    % Greedy Training hidden layers.
    for li=2:lAmt
        % Construct RBM from current layer of network.
        rbm=struct();
        rbm.W=NN{li}.W;
        rbm.vB=NN{li-1}.B;
        rbm.hB=NN{li}.B;
        
        % Training of RBM.
        rbm=rbm_train(rbm,X,opt);
        
        % Update RBM parameters to network.
        NN{li}.W=rbm.W;
        NN{li-1}.B=rbm.vB;
        NN{li}.B=rbm.hB;
        
        % Propagate input to next layer.
        X=rbm_up(rbm,X,true);%<<< stochastic propagation.
    end
else
    error('unknown method');
end

end