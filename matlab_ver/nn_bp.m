%Error backprop

function NN=nn_bp(NN,varargin)
lAmt=numel(NN);
if ~isempty(varargin)
    activeI=varargin{1};
    bDrop=true;
else
    bDrop=false;
end

if bDrop
    for li=lAmt-1:-1:2
        temp=NN{li+1}.err(:,activeI{li+1}) * NN{li+1}.W(activeI{li},activeI{li+1})';
        temp2=NN{li}.df(NN{li}.zin(:,activeI{li}));
        temp2(isnan(temp2))=0;
        NN{li}.err(:,activeI{li})=temp .* temp2;
    end
else
    for li=lAmt-1:-1:2
        temp=NN{li+1}.err * NN{li+1}.W';
        temp2=NN{li}.df(NN{li}.zin); 
        temp2(isnan(temp2))=0;%<--note1
        NN{li}.err=temp .* temp2;
    end
end
end
% note1: 由于多数activation function涉及指数项,当输入数据极小时，其导数的计算可能出现两Inf
% 变量相除的情况，尽管正确结果应接近0，但数字计算的结果变成Nan。特加入此条目进行纠正。