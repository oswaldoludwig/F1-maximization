%Author: Oswaldo Ludwig

%This function returns the linear classifier that maximizes the F1 measure given the training data pair (X,y).
%For details on this algorithm see Section 5.2 of the paper:
%Ludwig, Oswaldo, et al. "Deep Embedding for Spatial Role Labeling." arXiv preprint arXiv:1603.08474 (2016).
%In case of publication using this code, cite the above paper.

function [w_optm,b_optm]=max_F1(X,y)

learning_rate=2e-3; %set here the learning rate
patience=5000; %set here the patience hype-parameter
max_grad=2; %set here the threshold for the gradient

[L,C]=size(X);

%This code works with y belonging to the set {-1,1}, not {0,1}: 
if min(y)==0
    y=2*y-1;
end

%starting the gradient optimization from the LDA solution:
[w,b]=train_LDA(X,y,1, 1e-4);

w_optm=w;
b_optm=b;

index_neg=y==-1;
index_neg=find(index_neg);

index_pos=y==1;
index_pos=find(index_pos);

Xneg=X(:,index_neg);
Xpos=X(:,index_pos);

%Plotting the real value of F1 and its approximation:
[L,Cneg]=size(Xneg);
[L,Cpos]=size(Xpos);
figure(1)
hold

%Starting the optimization by gradient (see derivatives in the paper):
k=0;
cont=0;
cont_patience=0;
max_F1=0;
while cont_patience<patience
    k=k+1;
    cont=cont+1;
    cont_patience=cont_patience+1;
    TP=sum(1+tanh(w*Xpos+b))/2;
    TN=sum(1-tanh(w*Xneg+b))/2;
    F1=2*TP/(C+TP-TN);
    
    dTPdw=zeros(L,1);
    for n=1:Cpos
        dTPdw=dTPdw+(1-(tanh(w*Xpos(:,n)+b)).^2).*Xpos(:,n);
    end
    dTPdw=dTPdw/2;
    dTNdw=zeros(L,1);
    for n=1:Cneg
        dTNdw=dTNdw-(1-(tanh(w*Xneg(:,n)+b)).^2).*Xneg(:,n);
    end
    dTNdw=dTNdw/2;
    
    dTPdb=sum((1-(tanh(w*Xpos+b)).^2),2)/2;
    dTNdb=-sum((1-(tanh(w*Xneg+b)).^2),2)/2;
  
    dF1dw=2*F1*(2*TP*(-C-TP+TN)^(-2)*(-dTPdw+dTNdw)+2*dTPdw*(C+TP-TN)^(-1));
    gradW=norm(dF1dw);
    if gradW>max_grad
        dF1dw=dF1dw/gradW*max_grad;
    end
    dF1db=2*F1*(2*TP*(-C-TP+TN)^(-2)*(-dTPdb+dTNdb)+2*dTPdb*(C+TP-TN)^(-1));
    gradb=norm(dF1db);
    if gradb>max_grad
        dF1db=dF1db/gradb*max_grad;
    end
    w=w+learning_rate*dF1dw';
    b=b+learning_rate*dF1db;
    
    [precision,recall,F1_real]=prec_recall_F1(w*X+b,y,0);
    
    %Plotting real value of F1 and its approximation at each 150 iterations:
    if cont>150
        plot(k,F1_real,'r.')
        plot(k,F1,'b.')
        
        title('Fine tuning the output layer (red dot = real F1, blue dot = its approximation)')
        xlabel(['iteration=',num2str(k)])
        ylabel(['F1=',num2str(F1_real)])
        
        pause(0.001)
        cont=0;
    end
    if F1_real>max_F1
        max_F1=F1_real;
        w_optm=w;
        b_optm=b;
        cont_patience=0;
    end
      
end
close(figure(1))  



