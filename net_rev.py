# reverse the network - for generative model
def net_rev(NN,NP):
lNum=length(NN)

NNR=cell(lNum,1)
for li in range(1,lNum-1): 
    # neuron: R layer k = layer lNum-k+1; W: R layer k = layer lNum-k+2
    NNR[li]=struct()
    NNR[li]['W']=NN[lNum-li+2]['W']'
    NNR[li]['B']=NN[lNum-li+1]['B']
    NNR[li]['bBias']=false
    NNR[li]['f']=NN[lNum-li+1]['f']

NNR[lNum]['W']=NN[2]['W']'
NNR[lNum]['B']=np.zeros(1,NP['layerSize'](1))
NNR[lNum]['bBias']=false
NNR[lNum]['f']=NN[lNum]['f']


return NNR