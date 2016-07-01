# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 21:11:48 2016
@author: wb
"""
import numpy as np
import dm

# net,opt=net_setup([5,2,3,1],opt,typeOut=0)
def net_setup(layerSizes,opt=None,typeOut=0):
    # default option
    if opt==None:
        opt={}

    if 'unitFunc' not in opt:
        opt['unitFunc']='tanh'
    if 'unitFunc_output' not in opt:
        opt['unitFunc_output']='tanh'
    if 'costFunc' not in opt:
        opt['costFunc']='mse'
    if 'bBias' not in opt:
        opt['bBias']=True

    if 'lr' not in opt:
        opt['lr']=0.1
    if 'batchSize' not in opt:
        opt['batchSize']=10
    if 'regular' not in opt:
        opt['regular']='' # no regularization
    if 'regr' not in opt:
        opt['rr']=0.05
    if 'momentum' not in opt: # momentum friction ration
        opt['momentum']=0.8
    else:
        opt['momentum']=beinrange(opt['momentum'],0,1)

    if 'epochNum' not in opt:
        opt['epochNum']=20

    if typeOut:
        return opt
    else:
        print(opt)

    # Proc
    net={}
    net['LayerSize'] = layerSizes # include input layer to layer structure.
    lAmt = len(layerSizes)

    # Arrange network layer by layer (start at 1st hidden layer)
    net['layer']=[0]*lAmt
    net['layer'][0]={}
    for li in range(1,lAmt):
        net['layer'][li]={}
        ### Set Learning property and status
        net['layer'][li]['lr'] = opt['lr']
#        net.layer[li].momentum = opts.momentum
        net['layer'][li]['err'] = np.zeros(net['LayerSize'][li])
        net['layer'][li]['zin'] = np.zeros(net['LayerSize'][li])
        net['layer'][li]['act'] = np.zeros(net['LayerSize'][li])

        ### Set Structure (conetection and bias)
        net['layer'][li]['W']  = np.random.rand(net['LayerSize'][li-1], net['LayerSize'][li])/10
        if opt['bBias']:
            net['layer'][li]['B']  = np.zeros(net['LayerSize'][li])

        ### Set unit activation function
        if li<lAmt: # hidden layers
            layerUnit=opt['unitFunc']
        else: # output layer.
            layerUnit=opt['unitFunc_output']

        if layerUnit=='sigmoid': # positive sigmoid
            net['layer'][li]['f'] = lambda x: 1/(1+np.exp(-x))
            net['layer'][li]['df'] = lambda x: np.exp(-x)/(1+np.exp(-x))**2
        elif layerUnit=='tanh': # negative-positive sigmoid
            net['layer'][li]['f'] = lambda x: np.tanh(x)
            net['layer'][li]['df'] = lambda x: 4*np.exp(-x)/(2+np.exp(2*x)+np.exp(-2*x)) #1-tanh(x)**2
            # net['layer'][li]['df'] = lambda x: max(0.2, 4./(2+exp(2*x)+exp(-2*x)))
        elif layerUnit=='relu': # rectified linear
            net['layer'][li]['f'] = lambda x: x*(x>0)
            net['layer'][li]['df'] = lambda x: (x>0)+0
        elif layerUnit=='linear':
            net['layer'][li]['f'] = lambda x: x
            net['layer'][li]['df'] = lambda x: 1
        elif layerUnit== 'softmax':
            net['layer'][li]['f'] = lambda x: unit_softmax(x)
            net['layer'][li]['df'] = lambda x: df_softmax(x)
        else:
            print('unknown unit activation function')
            return []

    ### State variables (activity and input etc)
    net['layer'][li]['vW']=np.zeros(net['sizes'](li-1), net['sizes'](li))
    net['layer'][li]['vB']=np.zeros(1,net['sizes'](li))
    net['layer'][li]['momentumW']=net['layer'][li]['vW']; net['layer'][li]['momentumB']=net['layer'][li]['vB']
    net['layer'][li]['act']=[]
    net['layer'][li]['zin']=[]
    net['layer'][li]['e']=[]


    ### Set cost function (derivative of cost function to activity of output layer)
    if opt['costFunc'] == 'mse':
        net['cost']=lambda p,y: (p-y)**2
        net['detCost']=lambda p,y: p-y; # derivative of cost to activity of output layer.
    elif opt['costFunc'] == 'xent' : # cross entropy
        net['cost']=lambda p,y: cost_xent(p,y)
        net['detCost']=lambda p,y: detcost_xent(p,y)
    else:
        error('unknown cost function')


    ### Regularization
    if opt['regular'] == '':
        net['regular']=false
    elif opt['regular'] == 'L1':
        net['regular']['zor'] = lambda x: np.sign(x)
        net['regular']['rr']=opt['rr']
    elif opt['regular'] == 'L2':
        net['regular']['zor'] = lambda x: x
        net['regular']['rr']=opt['rr']
    else:
        error('unknown regularizor')

    return net


def unit_softmax(X): # sample must be in row
    temp=np.exp(X)
    Y=bsxfun(@rdivide,temp,sum(temp,2))
    return Y

def df_softmax(X):
    return

def cost_xent(p,y):
    p=dm.beinrange(p,1e-5,1-1e-5)
    c=y.*log(p)+(1-y).*log(1-p)
    return c

def detcost_xent(p,y):
    p=dm.beinrange(p,1e-5,1-1e-5)
    c=y./p + (1-y)./(1-p)
    return c