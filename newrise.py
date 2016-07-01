# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 15:58:09 2016
@author: wb
"""
import numpy as np
import dm


def run(net,X=None):
    if not X:
        X=np.zeros(net.lsize)

def ff(NN,X):
    lN=len(NN)

    NN[0]['zin']=X
    NN[0]['act']=X
    for li in range(1,lN):
        X = np.dot(X,NN[li]['W'])
        X += NN[li]['B']
        NN[li]['zin']=X
        X = NN[li]['f'](X) # no need to use map(f, X) for a numpy array
        NN[li]['act']=X

    return X,NN

# network training by backprop.
#  NN=nn_train(NN,X,Y,opt)
def nn_train(NN,X,Y,opt):
    sAmt=X.shape[0]
    lAmt=len(NN)
    batchSize=opt['batchSize']

    erec=[]
    for ei in range(opt['epochNum']):
        # permutation for stochastic learning.
        sampI=np.random.permutation(sAmt)
        segm=dm.cutseg(sAmt,batchSize)
        batchNum=len(segm)

        berec=np.zeros(batchNum)
        for bi in range(batchNum):
            # batch samples for stochastic training.
            I=sampI[segm[bi][0]:segm[bi][1]]

            # Propagate to get output
            xp,NN=ff(NN,X[I,:])

            # Get error (derivative of cost function) at output layer
            tp=NN.detCost(xp,Y[I,:]) #/batchSize # dC/d(a_L), as a batch, sum them together.
            NN[lAmt-1]['err'] = tp * NN[lAmt-1]['df'](NN[lAmt-1]['zin'])  # derivative of activation function to L layer input.

            # record the cost
            berec[bi]=sum(sum(NN['cost'](xp,Y[I,:])))

            # err Backprop
            for li in range(lAmt-2,0,-1):
                tp=np.dot(NN[li+1]['err'], NN[li+1]['W'].transpose())
                NN[li]['err']=tp * NN[li]['df'](NN[li]['zin'])


            ### Update W and B
            for li in range(1,lAmt):
                # derivative
                NN[li]['vW'] = np.dot(NN[li-1]['act'].transpose(), NN[li]['err']) # lr* de/d(W_k)
                # regularization
                if NN['regular']:
                    NN[li]['vW'] += opt['rr']*NN['regular']['zor'](NN[li]['W'])

                # turn to momentum
                NN[li]['momentumW'] = opt['momentum']*NN[li]['momentumW'] - NN[li]['lr']*NN[li]['vW']
                # update
                NN[li]['W'] += NN[li]['momentumW']
                # ? absolute limit of parameter
    #             NN.layer{li}.W=beinrange(NN.layer{li}.W,-1,1)

                if NN[li]['bBias']:
                    # derivative
                    NN[li]['vB'] = np.sum(NN[li]['err'],0)
                    # regularization
                    if NN['regular']:
                        NN[li]['vB'] += opt['rr']*NN['regular']['zor'](NN[li]['B'])

                    # turn to momentum
                    NN[li]['momentumB'] = opt['momentum']*NN[li]['momentumB'] - NN[li]['lr']*NN[li]['vB']
                    # update
                    NN[li]['B'] += NN[li]['momentumB']
                    #
    #                 NN.layer{li}.B=beinrange(NN.layer{li}.B,-1,1)

        erec+=berec
        print('|')

    plot(erec)
    return NN


