# 1 layer RNN
# layerSize: [input connection num, unit num]
def rnncell_setup(layerSize):
opt=struct('unitFunc_output','tanh','bBias',false)
NN=net_setup(layerSize,opt)
NN=NN(2)

NN[1]['R'] = randn(layerSize(2), layerSize(2))/sqrt(layerSize(2))

# NN{1}.zin=zeros(stepNum,layerSize(2))
NN[1]['act']=np.zeros(1,layerSize(2))
# NN{1}.err=zeros(stepNum,layerSize(2))

return NN