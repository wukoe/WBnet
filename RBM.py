"""module for RBM running and training."""
import numpy as np
import dm


# Upward propagate to hidden(next) layer.
#   X = rbm_up(rbm, X, bStochastic)
# bStochastic controls whether use stochastic sampling or expectation.
def rbm_up(rbm, X, bStoc=True):
    sAmt = X.shape[0]  # row as unit
    hAmt = len(rbm['hB'])

    # Get probability of Hidden units (=1)
    ne = rbm['hB'] + np.dot(X, rbm['W'])  # obs> np.repeat(rbm['hB'],sAmt,1) + X*rbm['W']
    P = 1 / (1 + np.exp(-ne))

    if bStoc:  # use stochastic binary states for hidden unit.
        X = np.ones(sAmt, hAmt)
        X[np.random.rand(sAmt, hAmt) > P] = 0
        return X
    else:
        return P


# Downward propagate to visible layer.
#   rbm_down(rbm, H, bStoc)
def rbm_down(rbm, H, bStoc=True):
    sAmt = H.shape[0]  # row as unit
    vAmt = len(rbm['vB'])

    # Get probability of visible units (=1)
    ne = rbm['vB'] + np.dot(H, np.transpose(rbm['W']))  # the negative energy
    P = 1 / (1 + np.exp(-ne))

    if bStoc:
        # use stochastic binary states for hidden unit.
        H = np.ones(sAmt, vAmt)
        H[np.random.rand(sAmt, vAmt) > P] = 0
        return H
    else:
        return P


""" RBM training.
#   [rbm,recRE]=rbm_train(rbm,SD,opt)
# rbm struct: 'W':weigth matrix (direction:col=input node; row=hidden node)
# 'vB':bias of visible(input) nodes
# 'hb':bias of hidden nodes
# 'dW','dvB','dhB':latest update of 3.
# SD take value in {1,0}. of shape [ch X pts]
# ! suggest initiate rbm connection matrix randomly to break symmetry(avoid
# learn to be same hidden as data?).
# Implementation details according to [Hinton 2010, A Practical Guide to Training Restricted Boltzmann Machines]
"""


def rbm_train(rbm, SD, opt, out=1):
    maxStep = opt['maxStep']  # with mini-batch training, this number could be smaller than using full batch.
    lr = 0.1  # 0.1 for update
    cdk = 1  # CD-k number

    # Process
    sAmt, dim = SD.shape
    # * IM.W 's size is [len(hidden)*len(visible)]
    innum = len(rbm['vB'])
    hnum = len(rbm['hB'])
    # make sure input->hidden is column->row
    assert rbm['W'].shape == [innum, hnum], 'W dim not match input and hidden units'

    # Decide batch size. With random sampling, doesn't have to cover every
    # sample.
    batchSize = min(np.floor(sAmt / 3), 50)
    batchAmt = np.floor(sAmt / batchSize)
    batchI = dm.cutseg([0, sAmt - 1], batchSize)

    ### Iterative update by Contrastive Divergence steps
    recRE = np.zeros(batchAmt, maxStep)
    for cyci in range(maxStep):
        RE = np.zeros(sAmt, 1)

        # Permutate the samples' order. by permutation of sample order
        permI = np.random.permutation(sAmt)
        for bi in range(batchAmt):
            bd = SD[permI[batchI[bi, 0]:batchI[bi, 2]], :]

            ### For 1st step (specifically keep out of circle so we can save the Hidden units states (for update)
            # Get probability of Hidden units (=1)
            ne = rbm['hB'] + np.dot(bd, rbm['W'])
            p1 = 1. / (1 + np.exp(-ne))
            # in encoding phase, use stochastic binary, take samples (this
            # works as regularizor for whole system thus necessary)
            h1 = np.ones(batchSize, hnum)
            h1[np.random.rand(batchSize, hnum) > p1] = 0

            # Get probability of reconstructed Visible units (=1)
            ne = rbm['vB'] + np.dot(h1, np.transpose(rbm['W']))
            P = 1. / (1 + np.exp(-ne))
            # pass probability instead of stochastic sampling for reconstruction.
            # it avoid sampling noise, learn faster, only slight performance
            # drop if any. For the regularization purpose, being in one pass is
            # enough.
            X = P

            ### If cdk>1, then do the rest steps
            if cdk > 0:
                for k in range(1, cdk):
                    # Get probability of Hidden units (=1)
                    ne = rbm['hB'] + np.dot(X, rbm['W'])
                    P = 1. / (1 + np.exp(-ne))
                    # use stochastic binary states for hidden unit.(now hidden
                    # unit is driven by reconstruction rather than data in visible)
                    H = np.ones(batchSize, hnum)
                    H[np.random.rand(batchSize, hnum) > P] = 0

                    # Get probability of visible units(=1) and assign to X.
                    ne = rbm['vB'] + np.dot(H, np.transpose(rbm['W']))
                    X = 1. / (1 + np.exp(-ne))

            # Get probability of Hidden units, at last round. no need to do
            # binary sampling here either.
            ne = rbm['hB'] + np.dot(X, rbm['W'])
            P = 1. / (1 + np.exp(-ne))

            ### Update (only suitable for 1/0 binomial unit)
            rbm['dW'] = lr * (np.transpose(bd) * p1 - np.dot(np.transpose(X), P)) / batchSize  # h1x1-Q(hk=1|xk)xk. p1 or h1 all works.
            rbm['dvB'] = lr * sum(bd - X) / batchSize  # x1-xk
            rbm['dhB'] = lr * sum(p1 - P) / batchSize  # h1-Q(hk=1|xk)

            rbm['W'] += rbm['dW']
            rbm['vB'] += rbm['dvB']
            rbm['hB'] += rbm['dhB']

            # Track the reconstruction error
            recRE[bi, cyci] = np.mean(np.mean(abs(bd - X)))

        print('|', end='')

    if out == 1:
        return rbm
    elif out==2:
        return rbm, recRE
    else:
        return []
