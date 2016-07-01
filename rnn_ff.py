# process one time series sample at a time.
def rnn_ff(nnc,X,flagTrain):
[sAmt,dim]=size(X)

if flagTrain:
    nnc['zin']=np.zeros(sAmt,dim)

if nnc['bBias']:
    repB=nnc['B']
else: 
    repB=0

R=nnc['R']; W=nnc['W']
Y=np.zeros(sAmt,dim)

# Init hidden unit activity.
if isfield(nnc,'act'):
    act=nnc['act'](,:)
else: 
    act=np.zeros(1,dim)

# Prop.
if flagTrain:
    for si in range(sAmt): 
        zin = act*R + X(si,:)*W
        zin = zin + repB
        act = nnc['f'](zin)
        Y(si,:) = act
        nnc['zin'](si,:) = zin
    
else: 
    for si in range(sAmt): 
        zin = act*R + X(si,:)*W
        zin = zin + repB
        act = nnc['f'](zin)
        Y(si,:) = act
    


if flagTrain:
    nnc['act']=Y



return [Y,nnc]