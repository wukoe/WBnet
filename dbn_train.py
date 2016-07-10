# train DBN
import RBM
import numpy as np

def dbn_train(NN,SD,opt):
    lAmt=len(NN['layer'])
    if 'maxStep' not in opt:
        opt['maxStep']=50


    ### Train the 1st hidden layer(connected by input layer).
    # Construct RBM from current layer of network.
    rbm={}
    rbm['W']=NN['layer'][2]['W']
    rbm['vB']=np.zeros(1,NN['sizes'](1))
    rbm['hB']=NN['layer'][2]['B']

    # Training of RBM.
    rbm=RBM.rbm_train(rbm,SD,opt) # input data as training data for 1st RBM

    # Update RBM parameters to network.
    NN['layer'][2]['W']=rbm['W']
    NN['layer'][1]['B']=rbm['vB']
    NN['layer'][2]['B']=rbm['hB']

    # Propagate input to next layer.
    X=RBM.rbm_up(rbm,SD) #stochastic propagation.

    ### Train other hidden layers.
    for li in range(2,lAmt):
        # Construct RBM from current layer of network.
        rbm={}
        rbm['W']=NN['layer'][li]['W']
        rbm['vB']=NN['layer'][li-1]['B']
        rbm['hB']=NN['layer'][li]['B']

        # Training of RBM.
        rbm=RBM.rbm_train(rbm,X,opt)

        # Update RBM parameters to network.
        NN['layer'][li]['W']=rbm['W']
        NN['layer'][li-1]['B']=rbm['vB']
        NN['layer'][li]['B']=rbm['hB']

        # Propagate input to next layer.
        X=RBM.rbm_up(rbm,X,True) #stochastic propagation.


    return NN