# RBM training.
#   [rbm,recRE]=rbm_train(rbm,SD,opt)
# rbm struct: 'W':weigth matrix (direction:col=input node; row=hidden node)
# 'vB':bias of visible(input) nodes
# 'hb':bias of hidden nodes
# 'dW','dvB','dhB':latest update of 3.
# SD take value in {1,0}. of shape [ch X pts]
# ! suggest initiate rbm connection matrix randomly to break symmetry(avoid
# learn to be same hidden as data?).
# Implementation details according to [Hinton 2010, A Practical Guide to Training Restricted Boltzmann Machines]
def rbm_train(rbm,SD,opt):
maxStep=opt['maxStep']; # with mini-batch training, this number could be smaller than using full batch.
lr=0.1; # 0.1 for update
cdk=1; # CD-k number

# Process
[sAmt,dim]=size(SD)
# * IM.W 's size is [len(hidden)*len(visible)]
if isempty(rbm) : # Initialize model
    rbm=struct()
    innum=dim
    hnum=opt['hNum']
    # set hidden unit bias to 0.
    rbm['hB']=np.zeros(1,hnum)
    # set visible unit bias to log(p/(1-p)), p is mean firing rate of data.
    P=sum(SD)/sAmt
    P=beinrange(P,0.1,0.9); # p=0 or 1 will make following equation infinite.
    rbm['vB']=log(P./(1-P))
    # initialize W to be small random number.
    rbm['W']=randn(innum,hnum)/100
else:  # use specified.
    innum=length(rbm['vB'])
    hnum=length(rbm['hB'])
    # make sure input->hidden is column->row
    assert(isequal(size(rbm['W']),[innum,hnum]),'W dim not match input and hidden units')


# Decide batch size. With random sampling, doesn't have to cover every
# sample.
batchSize=min(floor(sAmt/3),50)
batchAmt=floor(sAmt/batchSize)
batchI=cutseg([1,sAmt],batchSize)

### Iterative update by Contrastive Divergence steps
recRE=np.zeros(batchAmt,maxStep)
for cyci in range(maxStep): 
    RE=np.zeros(sAmt,1)

    # Permutate the samples' order. by permutation of sample order
    permI=randperm(sAmt)
    for bi in range(batchAmt): 
        bd=SD(permI(batchI(bi,1):batchI(bi,2)),:)
        bsa=batchSize

        ### For 1st step (specifically keep out of circle so we can save the Hidden units states (for update)
        # Get probability of Hidden units (=1)
        ne= repmat(rbm['hB'],bsa,1) + bd*rbm['W']
        p1=1./(1+exp(-ne))
        # in encoding phase, use stochastic binary, take samples (this
        # works as regularizor for whole system thus necessary)
        h1=np.ones(bsa,hnum); h1(np.random.rand(bsa,hnum)>p1)=0

        # Get probability of reconstructed Visible units (=1)
        ne= repmat(rbm['vB'],bsa,1) + h1*rbm['W']'
        P=1./(1+exp(-ne))
        # pass probability instead of stochastic sampling for reconstruction.
        # it avoid sampling noise, learn faster, only slight performance
        # drop if any. For the regularization purpose, being in one pass is
        # enough.
        X=P

        ### If cdk>1, then do the rest steps
        if cdk>1:
            for k in range(1,cdk): 
                # Get probability of Hidden units (=1)
                ne= repmat(rbm['hB'],bsa,1) + X*rbm['W']
                P=1./(1+exp(-ne))
                # use stochastic binary states for hidden unit.(now hidden
                # unit is driven by reconstruction rather than data in visible)
                H=np.ones(bsa,hnum); H(np.random.rand(bsa,hnum)>P)=0

                # Get probability of visible units(=1) and assign to X.
                ne= repmat(rbm['vB'],bsa,1) + H*rbm['W']'
                X=1./(1+exp(-ne))
            
        

        # Get probability of Hidden units, at last round. no need to do
        # binary sampling here either.
        ne= repmat(rbm['hB'],bsa,1) + X*rbm['W']
        P=1./(1+exp(-ne))

        ### Update (only suitable for 1/0 binomial unit)
        rbm['dW']=lr*(bd'*p1-X'*P)/bsa;# h1x1-Q(hk=1|xk)xk. p1 or h1 all works.
        rbm['dvB']=lr*sum(bd-X)/bsa; # x1-xk
        rbm['dhB']=lr*sum(p1-P)/bsa; # h1-Q(hk=1|xk)

        rbm['W']=rbm['W']+rbm['dW']
        rbm['vB']=rbm['vB']+rbm['dvB']
        rbm['hB']=rbm['hB']+rbm['dhB']

        # Track the reconstruction error
        recRE(bi,cyci)=mean(mean(abs(bd-X)))
    
    fprintf('|')



return [rbm,recRE]