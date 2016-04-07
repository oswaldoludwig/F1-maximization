%This function returns the estimated output given the net parameters and the input matrix X. 
function [estimated]=sim_NN(W1,W2,b1,b2,X)
[L,Col]=size(X);

L2=max(size(b2));
estimated=zeros(L2,Col);
 
for k=1:Col
    estimated(:,k)=W2*tansig(W1*X(:,k)+b1)+b2;
end