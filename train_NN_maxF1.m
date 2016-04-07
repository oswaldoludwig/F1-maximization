%Author: Oswaldo Ludwig

%This function returns the parameters of an MLP with a single hidden layer, with nneu hidden neurons,
%and a linear output layer that maximizes the F1 measure given the training data pair (X,y).
%For details on this algorithm see Section 5.2 of the paper:

%Ludwig, Oswaldo, et al. "Deep Embedding for Spatial Role Labeling." arXiv preprint arXiv:1603.08474 (2016).

%In case of publication using this code, cite the above paper.


function [W1,W2,b1,b2,yh]=train_NN_maxF1(X,y,nneu)
%training a simple MLP with a single hidden layer, with nneu hidden neurons, and a linear output layer:
L=size(X,1);
[C,namost]=size(y);
net=newff(X,y,nneu,{'tansig' 'purelin'});
net.trainParam.max_fail=10;
initnw(net,1);
initnw(net,2);
net.trainFcn = 'traingdx';
%net.trainParam.epochs = 3000; %uncomment this line to set the number of epochs (if you need more than 1000)
[net,E] = train(net,X(1:L,:),y);
%retrieving the trained synaptic weights:
W1=net.IW{1,1};
b1=net.b{1,1};
W2=net.LW{2,1};
b2=net.b{2,1};

%retrieving the hidden output, yh, in response to X:
yh=zeros(nneu,namost);
y_est=zeros(C,namost);
for k=1:namost
    yh(:,k)=tansig(W1*X(:,k)+b1);
    y_est(:,k)=W2*yh(:,k)+b2;
end

%Evaluating the F1 before fine tune the output layer:
for k=1:C
    [precision,recall,F1]=prec_recall_F1(y_est(k,:),y(k,:),0);
    disp(['Before the fine tuning the F1 for the output ',num2str(k),' is ',num2str(F1)])
end

%Fine tuning the linear output layer to maximize the F1 measure of all the outputs:
for k=1:C
    [W2(k,:),b2(k,1)]=max_F1(yh,y(k,:));
end

%Evaluating the F1 after fine tune the output layer:
for k=1:namost
    y_est(:,k)=W2*yh(:,k)+b2;
end
for k=1:C
    [precision,recall,F1]=prec_recall_F1(y_est(k,:),y(k,:),0);
    disp(['After the fine tuning the F1 for the output ',num2str(k),' is ',num2str(F1)])
end
