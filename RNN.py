"""The Recurrent NN module"""
import numpy as np

# process one time series sample at a time.
def rnn_ff(NN, X, flagTrain=False):
    sAmt = X.shape[0]
    hNum = NN['W'].shape[1]  # number of hidden units.

    if flagTrain:
        NN['zin'] = np.zeros(sAmt, hNum)

    if NN['bBias']:
        repB = NN['B']
    else:
        repB = 0

    R = NN['R']
    W = NN['W']
    Y = np.zeros(sAmt, hNum)

    # Init hidden unit activity.
    if 'act' in NN:
        act = NN['act'][-1,:]
    else:
        act = np.zeros(1, hNum)

    # Prop.
    if flagTrain:
        for si in range(sAmt):
            zin = np.dot(act, R) + np.dot(X[si,:], W)
            zin = zin + repB
            act = NN['f'](zin)
            Y[si,:] = act
            NN['zin'][si,:] = zin

    else:
        for si in range(sAmt):
            zin = np.dot(act, R) + np.dot(X[si,:], W)
            zin = zin + repB
            act = NN['f'](zin)
            Y[si, :] = act

    # Update activity state.
    if flagTrain:
        NN['act'] = Y
    else:
        NN['act'] = act

    return [Y, NN]


# Train RNN
def rnn_train(NN,X,Y,opt):
    sAmt = X.shape[0]
    hNum = NN['W'].shape[1]

    # Propagate sample in RNN.
    [P,NN]=rnn_ff(NN,X,True)

    # Backprop prepare.
    Rt=np.transpose(NN['R']) # recurrent conn transpose
    dfzin=NN['df'](NN['zin'])
    # * lower and higher bound of df
    # dfzin=max(dfzin,0.01)
    # dfzin=min(dfzin,10)

    # Get err from output connection
    detcost=P-Y # cost dev (MSE cost)
    outerr = detcost * dfzin
    # outerr(1:end-1)=0

    # Do recurrent layer error backprop through time.
    NN['err']=np.zeros(sAmt,hNum)
    NN['err'][sAmt-1,:]=outerr[sAmt,:]
    for si in range(sAmt-1,0,-1):
        temp = outerr[si,:] + np.dot(NN['err'][si+1,:], Rt) # error from output(t) + from next step(t+1).
        NN['err'][si,:] = temp * dfzin(si,1); # error at this step. ??????
    #     temp = min(temp,5); for restrict the range of error.
    #     temp = max(temp,-5)


    # Update for the parameter.
    dR = np.dot(np.transpose(NN['act'][0:sAmt-1,:]), NN['err'][1:sAmt,:])
    dW = np.dot(np.transpose(X), NN['err'])
    NN['R'] = NN['R'] - opt['lr'] * dR #/(sAmt-1)
    NN['W'] = NN['W'] - opt['lr'] * dW #/sAmt
    if NN['bBias']:
        dB = sum(NN['err'],1)
        NN['B'] = NN['B'] - opt['lr'] * dB #/sAmt

    return NN,outerr