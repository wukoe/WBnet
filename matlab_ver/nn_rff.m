% reverse transmission of network
function X=nn_rff(NN,X)
lAmt=numel(NN);
sAmt=size(X,1);

X=beinrange(X,-0.99,0.99);
X=-log(1./X-1); %atanh(X);
X = X - repmat(NN{lAmt}.B, sAmt,1);
X = X * NN{lAmt}.W';
for li=lAmt-1:-1:2
    X=beinrange(X,-0.99,0.99);
    X=atanh(X);
    if NN{li}.bBias
        X = X - repmat(NN{li}.B, sAmt,1);
    end
    X = X * NN{li}.W'; % practically much better than "pinv(W)"
end

end