# layer 0 same size as input data, specify neuron numbers in each
# additional layer.
# Definition rules: W from layer i to i+1 is attached to layer i+1. In
# other words, each layer has unified structure: W,b->z-(f)->a
def net_setup(inDim,layerSizes,opt):
### Default network options.
# structure
if ~isfield(opt,'unitFunc'):
    opt['unitFunc']='tanh'

if ~isfield(opt,'unitFunc_output'):
    opt['unitFunc_output']='tanh'

if ~isfield(opt,'costFunc'):
    opt['costFunc']='mse'

if ~isfield(opt,'bBias'):
    opt['bBias']=true


# learning rate
if ~isfield(opt,'lr'):
    opt['lr']=0.1

if ~isfield(opt,'batchSize'):
    opt['batchSize']=10

if ~isfield(opt,'epochAmt'):
    opt['epochAmt']=1


# learning optimization
if ~isfield(opt,'regular') || isempty(opt['regular']):
    opt['regular']=''

if ~isfield(opt,'rr'):
    opt['rr']=0.05

if ~isfield(opt,'momentum'):
    opt['momentum']=0.8;# momentum friction ration
else: 
    opt['momentum']=beinrange(opt['momentum'],0,1)

if ~isfield(opt,'dropr'):
    opt['dropr']=0;# dropout rate

disp(opt)

### Arrange network
lAmt = numel(layerSizes)+1
NP=struct(); # network training structures.
NN=cell(lAmt,1); # network layer structures.
NP['inDim'] = inDim
NP['layerSize'] = [NP['inDim'], layerSizes]; # include input layer to layer structure.

# Set Learning property (for each layer)
if numel(opt['lr'])==lAmt:
    NP['lr']=opt['lr']
else: 
    NP['lr']=np.zeros(lAmt,1)
    for li in range(1,lAmt): 
        NP['lr'](li) = opt['lr']
    

for li in range(1,lAmt): 
    NN[li]['bBias']=opt['bBias']; # all layers use same setting


# Layers.
NN[1]['act']=[]
for li in range(1,lAmt): 
    ### Set Structure (connection and bias)
    # * important to initialize with Gaussian with STD reversely fit to
    # size of previous layer.
    NN[li]['W']  = randn(NP['layerSize'](li-1), NP['layerSize'](li))/sqrt(NP['layerSize'](li-1))
    if NN[li]['bBias']:
        NN[li]['B']  = np.zeros(1,NP['layerSize'](li))
    

    ### Set unit activation function
    if li<lAmt : # hidden layers
        layerUnit=opt['unitFunc']
    else:  # output layer.
        layerUnit=opt['unitFunc_output']
    
    # * XXXX建议不要在此处进行df的阈值设定，到backprop处进行clipping处理。XXXX
    
        elif layerUnit== 'sigmoid' : # positive sigmoid
            NN[li]['f'] = @(x)1./(1+exp(-x))
            NN[li]['df'] = @(x)exp(-x)./(1+exp(-x)).**2
#             NN{li}.df = @(x)max(0.025,exp(-x)./(1+exp(-x)).^2)
        elif layerUnit== 'tanh' : # negative-positive sigmoid
            NN[li]['f'] = @(x)tanh(x)
            NN[li]['df'] = @(x)4./(2+exp(2*x)+exp(-2*x)); # 1-tanh(x)**2
#             NN{li}.df = @(x)max(0.1, 4./(2+exp(2*x)+exp(-2*x)))
#             NN{li}.df = @(x)0.25
        elif layerUnit== 'relu':
            NN[li]['f'] = @(x)x.*(x>0)
            NN[li]['df'] = @(x)(x>0)
        elif layerUnit== 'linear':
            NN[li]['f'] = @(x)x
            NN[li]['df'] = @(x)1
        elif layerUnit== 'softmax':
            NN[li]['f']=@unit_softmax
            NN[li]['df']=@df_softmax
        else:
            error('unknown unit activation def ):
    

    ### State variables (activity and input etc)
    NN[li]['vW']=np.zeros(NP['layerSize'](li-1), NP['layerSize'](li))
    NN[li]['momentumW']=NN[li]['vW']
    if NN[li]['bBias']:
        NN[li]['vB']=np.zeros(1,NP['layerSize'](li))
        NN[li]['momentumB']=NN[li]['vB']
    
    NN[li]['act']=[]
    NN[li]['zin']=[]
    NN[li]['e']=[]


### Set cost function (derivative of cost function to activity of output layer)
['costFunc']
    elif opt== 'mse':
        NP['cost']=@(p,y)sum((p-y).**2,2)/2
        NP['detCost']=@(p,y)p-y; # derivative of cost to activity of output layer.
    elif opt== 'xent' : # cross entropy
        NP['cost']=@cost_xent
        NP['detCost']=@detcost_xent
    else:
        error('unknown cost def ):


### Regularization
if ~isequal(opt['regular'],''):
    ['regular']
        elif opt== 'L1':
            NP['regzor'] = @(x)sign(x)
        elif opt== 'L2':
            NP['regzor'] = @(x)x
        else:
            error('unknown regularizor')
    




### function of units
def unit_softmax(X): # sample must be in row
temp=exp(X)
Y=bsxfun(@rdivide,temp,sum(temp,2))

# softmax have mixing, so its det is different from others, take 2 input.
def df_softmax(Y,detcost):
[sa,dim]=size(Y)
E=np.zeros(sa,dim)
for k in range(sa): 
    D=diag(Y(k,:))-Y(k,:)'*Y(k,:); # derivative of sample k.
    E(k,:)=detcost(k,:)*D



### function of cost.
# This cross entropy is for 2 class label classification, and regression.
def cost_xent(p,y):
p=beinrange(p,1e-5,1-1e-5)
c=-(y.*log(p)+(1-y).*log(1-p))
c=sum(c,2)

def detcost_xent(p,y):
p=beinrange(p,1e-5,1-1e-5)
c=-y./p+(1-y)./(1-p)

return c