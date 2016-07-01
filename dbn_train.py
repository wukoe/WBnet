# train DBN
def dbn_train(NN,SD,opt):
lAmt=numel(NN['layer'])
if ~isfield(opt,'maxStep'):
    opt['maxStep']=50


### Train the 1st hidden layer(connected by input layer).
# Construct RBM from current layer of network.
rbm=struct()
rbm['W']=NN['layer'][2]['W']
rbm['vB']=np.zeros(1,NN['sizes'](1))
rbm['hB']=NN['layer'][2]['B']

# Training of RBM.
rbm=rbm_train(rbm,SD,opt);# input data as training data for 1st RBM

# Update RBM parameters to network.
NN['layer'][2]['W']=rbm['W']
NN['layer'][1]['B']=rbm['vB']
NN['layer'][2]['B']=rbm['hB']

# Propagate input to next layer.
X=rbm_up(rbm,SD,true);#stochastic propagation.

### Train other hidden layers.
for li in range(2,lAmt): 
    # Construct RBM from current layer of network.
    rbm=struct()
    rbm['W']=NN['layer'][li]['W']
    rbm['vB']=NN['layer'][li-1]['B']
    rbm['hB']=NN['layer'][li]['B']

    # Training of RBM.
    rbm=rbm_train(rbm,X,opt)

    # Update RBM parameters to network.
    NN['layer'][li]['W']=rbm['W']
    NN['layer'][li-1]['B']=rbm['vB']
    NN['layer'][li]['B']=rbm['hB']

    # Propagate input to next layer.
    X=rbm_up(rbm,X,true);#stochastic propagation.



return NN