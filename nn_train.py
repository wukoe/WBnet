
# network training by backprop.
#   [NN,erec]=nn_train(NN,NP,X,Y,opt)
def nn_train(NN,NP,X,Y,opt,varargin):
    bAddUnsErr=false; # if add unsupervised learning error.

    ### Proc
    sAmt=size(X,1)
    lAmt=len(NN)
    [segm,batchAmt]=cutseg([1,sAmt],opt['batchSize'],'equal')
    # validation set
    if ~isempty(varargin):
        flagValidset=true
        validX=varargin[1]
        validY=varargin[2]
    else:
        flagValidset=false


    # Running option
    bReg=~isequal(opt['regular'],''); # regularization
    if opt['dropr']>0:
        bDrop=true
        activeI=cell(lAmt,1)
        activeI[1]=true(1,NP['layerSize'](1))
        activeI[lAmt]=true(1,NP['layerSize'](lAmt))
    else:
        bDrop=false


    if bAddUnsErr:
        # fix missing items for local AE backward signal.
        for li in range(lAmt-2):
            if ~isfield(NN[li],'rB'):
                NN[li]['rB']=np.zeros(1,NP['layerSize'](li))

            if ~isfield(NN[li],'momentumrB'):
                NN[li]['momentumrB']=np.zeros(1,NP['layerSize'](li))


        NN[1]['f']=NN[2]['f']
        NN[1]['df']=NN[2]['df']
        tpopt=struct('costFunc','mse')
        [~,rnnp]=net_setup([1,1,1],tpopt)



    ###
    trerr=np.zeros(batchAmt*opt['epochAmt'],1); vaerr=trerr
    ct=0
    for ei in range(opt['epochAmt']):
        # permutation for stochastic learning.
        sampI=randperm(sAmt)

        for bi in range(batchAmt):
            ### batch picking
            I=sampI(segm(bi,1):segm(bi,2)); # current batch for stochastic training.
            batchSize=length(I)
            ct=ct+1
            # Drop-out filters
            if bDrop:
                for li in range(1,lAmt-1):
                    tp=np.random.rand(1,NP['layerSize'](li))
                    activeI[li]=tp>opt['dropr']



            ### Forward path & Backprop
            if bDrop:
                [bp,NN]=nn_ff(NN,X(I,:),true,activeI); # bp=batch prediction
            else:
                [bp,NN]=nn_ff(NN,X(I,:),true)


            # Get training cost
            trerr(ct)=mean(NP['cost'](bp,Y(I,:))); # cost at output layer
            # Get validation cost
            if flagValidset:
                tp=nn_ff(NN,validX)
                vaerr(ct)=mean(NP['cost'](tp,validY))


            # Get cost derivative (error) at output layer
            detcost=NP['detCost'](bp,Y(I,:));# dC/d(a_L), do NOT sum batch together - summing happen at vW.
            # Error at output layer
            if nargin(NN[lAmt]['df'])==1:
                temp=NN[lAmt]['df'](NN[lAmt]['zin']); # derivative of final output to output layer in-sum.
                NN[lAmt]['err'] = detcost .* temp
            else:  # if f'() of output have 2 input term - mostly for softmax layer.
                NN[lAmt]['err'] = NN[lAmt]['df'](bp,detcost)


            # Error Backprop
            NN=nn_bp(NN)

    #         ### Gradiant check. (actual error VS predicted error)
    #         de=1e-5
    #         chklayer=5; chkw=[10,4]
    #         nn=NN
    #         nn{chklayer}.W(chkw(1),chkw(2))=nn{chklayer}.W(chkw(1),chkw(2))+de
    #         [tpbp]=nn_ff(nn,X(I,:))
    #         tpc1=NP.cost(tpbp,Y(I,:))
    #         nn{chklayer}.W=NN{chklayer}.W
    #         nn{chklayer}.W(chkw(1),chkw(2))=nn{chklayer}.W(chkw(1),chkw(2))-de
    #         [tpbp]=nn_ff(nn,X(I,:))
    #         tpc2=NP.cost(tpbp,Y(I,:))
    #
    #         cdev=(tpc1-tpc2)/(2*de)
    #         dW=NN{chklayer-1}.act(:,chkw(1)) .* NN{chklayer}.err(:,chkw(2))
    #         gap=cdev-dW
    #
    #         cla
    #         hold on
    #         plot(cdev,'r')
    #         plot(dW)

            ### Add the unsupervised learning error.
            if bAddUnsErr:
                for li in range(1,lAmt-1):
                    # Reverse FF
                    rzin = NN[li]['act'] * NN[li]['W']'
                    rzin = rzin + repmat(NN[li-1]['rB'],batchSize,1)
                    ry = NN[li-1]['f'](rzin)

                    # Cost and error
                    rdetcost=rnnp['detCost'](ry,NN[li-1]['act'])
                    temp=NN[li-1]['df'](rzin); # derivative of reconstruction to in-sum
                    NN[li-1]['rerr'] = rdetcost .* temp; # error at reconstruct layer


    #             # Match uns error to size of BP error.
    #             # set base value
    #             basev=mean(mean(abs(NN{lAmt-1}.err)))
    #             #
    #             for li=2:lAmt-1
    #                 tp=mean(mean(abs(NN{li}.err)))
    #                 tpr=mean(mean(abs(NN{li-1}.rerr)))
    #                 if tp+tpr>basev
    #                     NN{li-1}.rerr=NN{li-1}.rerr*(basev-tp)/tpr
    #                 end
    #             end


            ### Update W and B
    #         tp2=zeros(lAmt,1); # for change of params
            if bDrop:
                for li in range(lAmt,-1:2):
                    # derivative
                    vW(activeI[li-1],activeI[li]) = NN[li-1]['act'](:,activeI[li-1])' * NN[li]['err'](:,activeI[li]);# =de/d(W_k)
                    # regularization
                    if bReg:
                        vW(activeI[li-1],activeI[li]) = vW(activeI[li-1],activeI[li]) + opt['rr']*NP['regzor'](NN[li]['W'](activeI[li-1],activeI[li]))

                    # turn to momentum
                    NN[li]['momentumW'](activeI[li-1],activeI[li]) = opt['momentum']*NN[li]['momentumW'](activeI[li-1],activeI[li]) - NP['lr'](li)*vW(activeI[li-1],activeI[li])
                    # update
                    NN[li]['W'](activeI[li-1],activeI[li]) = NN[li]['W'](activeI[li-1],activeI[li]) + NN[li]['momentumW'](activeI[li-1],activeI[li])
                    # ? absolute limit of parameter
                    #             NN{li}.W=beinrange(NN{li}.W,-1,1)

                    if NN[li]['bBias']:
                        # derivative
                        vB(activeI[li]) = sum(NN[li]['err'](:,activeI[li]),1)
                        # regularization
                        if bReg:
                            vB(activeI[li]) = vB(activeI[li]) + opt['rr']*NP['regzor'](NN[li]['B'](activeI[li]))

                        # turn to momentum
                        NN[li]['momentumB'](activeI[li]) = opt['momentum']*NN[li]['momentumB'](activeI[li]) - NP['lr'](li)*vB(activeI[li])
                        # update
                        NN[li]['B'](activeI[li]) = NN[li]['B'](activeI[li]) + NN[li]['momentumB'](activeI[li])
                        #                 NN{li}.B=beinrange(NN{li}.B,-1,1)


            else:
                for li in range(lAmt,-1:2):
                    # derivative
                    vW = NN[li-1]['act']' * NN[li]['err'];# =de/d(W_k)
                    # add reverse derivative
                    if bAddUnsErr:
                        if li==lAmt:
                            basev=mean(mean(abs(vW)))
                        else:  #li<lAmt
                            temp = NN[li]['act']' * NN[li-1]['rerr']

                            # Match uns dW to size of FF dW.
                            tp=mean(mean(abs(vW)))
                            tpr=mean(mean(abs(temp)))
                            if tp+tpr>basev:
                                temp=temp*(basev-tp)/tpr


                            vW = vW + temp'
    #                         tp2(li)=mean(mean(abs(vW)))


                    # add regularization
                    if bReg:
                        vW = vW + opt['rr']*NP['regzor'](NN[li]['W'])

                    # turn to momentum
                    NN[li]['momentumW'] = opt['momentum']*NN[li]['momentumW'] - NP['lr'](li)*vW
                    # update
                    NN[li]['W'] = NN[li]['W'] + NN[li]['momentumW']
                    # ? absolute limit of parameter
                    # NN{li}.W=beinrange(NN{li}.W,-1,1)

                    if NN[li]['bBias']:
                        # derivative
                        vB = sum(NN[li]['err'],1)
                        # regularization
                        if bReg:
                            vB = vB + opt['rr']*NP['regzor'](NN[li]['B'])

                        # turn to momentum
                        NN[li]['momentumB'] = opt['momentum']*NN[li]['momentumB'] - NP['lr'](li)*vB
                        # update
                        NN[li]['B'] = NN[li]['B'] + NN[li]['momentumB']
                        #  NN{li}.B=beinrange(NN{li}.B,-1,1)

                        # rB and B is different, use separate processing.
                        if bAddUnsErr && li<lAmt-1:
                            vrB=sum(NN[li]['rerr'],1)
                            if bReg:
                                vrB = vrB + opt['rr']*NP['regzor'](NN[li]['rB'])

                            NN[li]['momentumrB'] = opt['momentum']*NN[li]['momentumrB'] - NP['lr'](li)*vrB
                            # update
                            NN[li]['rB'] = NN[li]['rB'] + NN[li]['momentumrB']







            # <<< disp error strength of very layer
    #         tp1=zeros(lAmt,1); # for error
    #         for li=1:lAmt-2
    #             temp=mean(abs(NN{li}.err))
    #             tp1(li)=mean(temp)
    #         end
    #         plot([tp2])

            ### Adjust lr
    #         lrrec(idx)=NN{2}.lr
    #         # monistering module
    #         if idx>99 && mod(bi,100)==0
    #             temp=erec(idx-49:idx)
    #             plot(temp)
    #             tp1=(mean(temp(1:10))-mean(temp(end-9:end))); tp2=std(temp)
    #             if tp2/tp1>5 && tp2/mean(temp)>10 #|| tp1<0
    #                 NP=netlr(NP,max(NP.lr(2)/2,0.00001))
    #             end
    #         end

    #         ### Adjust error strength (结论：也许不必要)
    #         mw=zeros(lAmt,1)
    #         for li=2:lAmt
    #             mw(li)=mean(mean(abs(NN{li}.err)))
    #         end
    #         for li=lAmt-1:-1:2
    #             NN{li}.err=NN{li}.err*mw(lAmt)/mw(li)
    #         end

        fprintf('|')

    fprintf('\n')

    if flagValidset:
        plot([trerr,vaerr])
    else:
        plot(trerr)



    return [NN,trerr]