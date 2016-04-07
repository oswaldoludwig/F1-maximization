function [w,b]=train_LDA(X,y,alpha, regula)

index_neg=y==-1;
index_neg=find(index_neg);
index_pos=y==1;
index_pos=find(index_pos);
L=size(X,1);
featneg = mean(X(:,index_neg),2);
featpos = mean(X(:,index_pos),2);
w = (featpos - featneg)'/ ( alpha * cov(X(:,index_neg)') + (1-alpha) * cov(X(:,index_pos)') + regula* eye(L))';
b = - 1/2 * (w * (featneg + featpos));

