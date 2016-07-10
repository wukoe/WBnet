"""Feed forward net"""

# Feed-forward network, forward propagation.
#   [X,NN]=nn_ff(NN,X,varargin) varin=I
def fnn_ff(NN, X, flagTrain, activeI=None):
    lAmt = len(NN)
    sAmt = X.shape[0]

    if activeI is None:
        bDrop = False
    else:
        bDrop = True

    # Input layer
    NN[0]['act'] = X
    if bDrop:
        for li in range(1, lAmt):
            X = X * NN[li]['W'](activeI[li - 1], activeI[li])
            if NN[li]['bBias']:
                X = X + repmat(NN[li]['B'](activeI[li]), sAmt, 1)

            NN[li]['zin'](:, activeI[li])=X;  # total input is necessary to record too.
            X = NN[li]['f'](X);  # * another form arrayfun(@f,X)
            NN[li]['act'](:, activeI[li])=X

    else:
        for li in range(1, lAmt):
            X = X * NN[li]['W']
            if NN[li]['bBias']:
                X = X + repmat(NN[li]['B'], sAmt, 1)

            if flagTrain:
                NN[li]['zin'] = X;  # total input is necessary to record too.
                X = NN[li]['f'](X);  # * another form arrayfun(@f,X)
                NN[li]['act'] = X
            else:
                X = NN[li]['f'](X);  # * another form arrayfun(@f,X)
                NN[li]['zin'] = []
                NN[li]['act'] = []
                NN[li]['err'] = []
                NN[li]['vW'] = [];
                NN[li]['momentumW'] = []
                NN[li]['vB'] = [];
                NN[li]['momentumB'] = []

    return [X, NN]