% 1 layer RNN
% layerSize: [input connection num of units, hidden unit num]
% recurrent connection is default [hidden X hidden].
function NN=rnncell_setup(layerSize)
opt=struct('unitFunc_output','tanh','bBias',false);
NN=net_setup(layerSize,opt);
NN=NN(2);

NN{1}.R = randn(layerSize(2), layerSize(2))/sqrt(layerSize(2));

% NN{1}.zin=zeros(stepNum,layerSize(2));
NN{1}.act=zeros(1,layerSize(2));
% NN{1}.err=zeros(stepNum,layerSize(2));
end