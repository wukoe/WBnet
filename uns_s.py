# using unsupervised pre-training
#   [NN,err] = uns_s(NN,NP,X,Y,opt)
def uns_s(NN,NP,X,Y,opt):
lNum=length(NN)

dbnLastLayer=4; # training by DNB of part layers end at this.
bpFirstLayer=4; # DP training part layers, start at this. (bpFirstLayer as input layer, not to be trained)

### Unsupervised Training of network by DBN.
optdbn=opt
optdbn['unitFunc_output']=optdbn['unitFunc']
NN(1:dbnLastLayer)=dbn_train(NN(1:dbnLastLayer),NP,X,'sae',optdbn)

### BP training.
if bpFirstLayer>1:
    tx=nn_ff(NN(1:bpFirstLayer),X)
    np=NP
    np['inDim']=NP['layerSize'](bpFirstLayer)
    np['layerSize']=NP['layerSize'](bpFirstLayer:lNum)
    np['lr']=[0,NP['lr'](bpFirstLayer+1:lNum)]
    [NN(bpFirstLayer:lNum),err]=nn_train(NN(bpFirstLayer:lNum),np,tx,Y,opt)
else: 
    [NN,err]=nn_train(NN,NP,X,Y,opt)



return [NN,err]