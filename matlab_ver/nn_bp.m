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
% note1: ���ڶ���activation function�漰ָ����,���������ݼ�Сʱ���䵼���ļ�����ܳ�����Inf
% ��������������������ȷ���Ӧ�ӽ�0�������ּ���Ľ�����Nan���ؼ������Ŀ���о�����