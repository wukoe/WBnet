% Materialize a network.
%   [NN,NP,opt] = net_setup(layerSizes,opt)
% layerSizes specify neuron numbers in each layer. layerSizes(1) is input data
% Definition rules: W from layer i to i+1 is attached to layer i+1. In
% other words, each layer has unified structure: W,b->z-(f)->a
function [NN,NP,opt] = net_setup(layerSize,opt)
lAmt = numel(layerSize);

%%% Default network options.
% structure
if ~isfield(opt,'unitFunc')
    opt.unitFunc='tanh';
end
if ~isfield(opt,'unitFunc_output')
    opt.unitFunc_output='sigmoid';
end
if ~isfield(opt,'costFunc')
    opt.costFunc='xent';
end
if ~isfield(opt,'bBias')
    opt.bBias=true;
end

% learning rate
if ~isfield(opt,'lr') || strcmp(opt.lr,'auto')
    opt.lr=[0, 1./layerSize(1:lAmt-1)];
    %*按照每神经元接受的总学习量为1似乎不错。
end
if ~isfield(opt,'batchSize')
    opt.batchSize=10;
end
if ~isfield(opt,'epochAmt')
    opt.epochAmt=1;
end

% learning optimization
if ~isfield(opt,'regular') || isempty(opt.regular)
    opt.regular='';
end
if ~isfield(opt,'rr')
    opt.rr=0.05;
end
if ~isfield(opt,'momentum')
    opt.momentum=0.8;% momentum friction ration
else
    opt.momentum=beinrange(opt.momentum,0,1);
end
if ~isfield(opt,'dropr')
    opt.dropr=0;% dropout rate
end

%%% Arrange network
NP=struct(); % network training structures.
NN=cell(lAmt,1); % network layer structures.
NP.inDim = layerSize(1);
NP.layerSize = layerSize; % include input layer to layer structure.

% Set Learning property (for each layer)
if numel(opt.lr)==lAmt
    NP.lr=opt.lr;
else
    NP.lr=zeros(lAmt,1);
    for li=2:lAmt
        NP.lr(li) = opt.lr;
    end
end
% initialize layer struct
NN{1}=struct();
NN{1}.act=[];
for li=2:lAmt
    NN{li}=struct();
    NN{li}.bBias=opt.bBias; % all layers use same setting
end

% Layers.
for li = 2 : lAmt    
    %%% Set Structure (connection and bias)
    % * important to initialize with Gaussian with STD reversely fit to
    % size of previous layer.
    NN{li}.W = randn(NP.layerSize(li-1), NP.layerSize(li))/sqrt(NP.layerSize(li-1));
    if NN{li}.bBias
        NN{li}.B = zeros(1,NP.layerSize(li));
    end
%     use a different backprop weight.
%     NN{li}.rW=randn(NP.layerSize(li-1), NP.layerSize(li))'/sqrt(NP.layerSize(li-1));
    
    %%% Set unit activation function
    if li<lAmt % hidden layers
        layerUnit=opt.unitFunc;
    else % output layer.
        layerUnit=opt.unitFunc_output;
    end
    % * XXXX建议不要在此处进行df的阈值设定，到backprop处进行clipping处理。XXXX
    switch layerUnit
        case 'sigmoid' % positive sigmoid
            NN{li}.f = @(x)1./(1+exp(-x));
            NN{li}.df = @(x)exp(-x)./(1+exp(-x)).^2;
%             NN{li}.df = @(x)max(0.1,exp(-x)./(1+exp(-x)).^2);
        case 'tanh' % negative-positive sigmoid
            NN{li}.f = @(x)tanh(x);
            NN{li}.df = @(x)4./(2+exp(2*x)+exp(-2*x)); % 1-tanh(x)^2
%             NN{li}.df = @(x)max(0.1, 4./(2+exp(2*x)+exp(-2*x)));
        case 'relu'
            NN{li}.f = @(x)x.*(x>0);
            NN{li}.df = @(x)(x>0);
        case 'relu1'% at negative, has a slant of 0.1
            NN{li}.f = @(x)x.*((x>0)*0.9+0.1);
            NN{li}.df = @(x)(x>0)*0.9+0.1;
        case 'linear'
            NN{li}.f = @(x)x;
            NN{li}.df = @(x)1;
        case 'softmax'
            NN{li}.f=@unit_softmax;
            NN{li}.df=@df_softmax;
        case 'lseg1' % local linear segmen 1 (my own test)
            NN{li}.f=@unit_lseg1;
            NN{li}.df=@df_lseg1;
        case 'lseg2' % local linear segmen 2 (my own test)
            NN{li}.f=@unit_lseg2;
            NN{li}.df=@df_lseg2;
        otherwise
            error('unknown unit activation function');
    end    
    
    %%% State variables (activity and input etc)   
    NN{li}.vW=zeros(NP.layerSize(li-1), NP.layerSize(li));
    NN{li}.momentumW=NN{li}.vW;
    if NN{li}.bBias
        NN{li}.vB=zeros(1,NP.layerSize(li));
        NN{li}.momentumB=NN{li}.vB;
    end
    NN{li}.act=[];
    NN{li}.zin=[];
    NN{li}.err=[];
end

%%% Set cost function (derivative of cost function to activity of output layer)
switch opt.costFunc
    case 'mse'
        NP.cost=@(p,y)sum((p-y).^2,2)/2;
        NP.detCost=@(p,y)p-y; % derivative of cost to activity of output layer.
    case 'xent' % cross entropy
        NP.cost=@cost_xent;
        NP.detCost=@detcost_xent;
    case 'likelihood' % negative likelihood - 注意一般只适用于和输出在(0,1) 如sigmoid
    otherwise
        error('unknown cost function');
end

%%% Regularization
if ~isequal(opt.regular,'')
    switch opt.regular
        case 'L1'
            NP.regzor = @(x)sign(x);
        case 'L2'
            NP.regzor = @(x)x;
        otherwise
            error('unknown regularizor');
    end
end

end

%%% function of units
function Y=unit_softmax(X) % sample must be in row
temp=exp(X);
Y=bsxfun(@rdivide,temp,sum(temp,2));
end
% softmax have mixing, so its det is different from others, take 2 input.
function E=df_softmax(Y,detcost)
[sa,dim]=size(Y);
E=zeros(sa,dim);
for k=1:sa
    D=diag(Y(k,:))-Y(k,:)'*Y(k,:); % derivative of sample k.
    E(k,:)=detcost(k,:)*D;
end
end

function Y=unit_lseg1(X)
Y=X; % for X in [-1,1]
I=X>2;
Y(I)=2;%1.8+0.1*Y(I);
I=X<-2;
Y(I)=-2;%-1.8+0.1*Y(I);
end
function Y=df_lseg1(X)
Y=ones(size(X)); % for X in [-1,1]
I=(X>2)|(X<-2);
Y(I)=0.05;
end

function Y=unit_lseg2(X)
Y=X; % for X in [-1,1]
I=X>5;
Y(I)=4.5+0.1*Y(I);
I=X<0;
Y(I)=0.1*Y(I);
end
function Y=df_lseg2(X)
Y=ones(size(X)); % for X in [-1,1]
I=(X>5)|(X<0);
Y(I)=0.1;
end


%%% function of cost.
% This cross entropy is for 2 class label classification, and regression.
function c=cost_xent(p,y) 
p=beinrange(p,1e-5,1-1e-5);
c=-(y.*log(p)+(1-y).*log(1-p));
c=sum(c,2);
end
function c=detcost_xent(p,y)
p=beinrange(p,1e-5,1-1e-5);
c=-y./p+(1-y)./(1-p);
end