%This function evaluates the precision, recall and F1 measures, given the estimated and target values, y and y_targ, and the threshold, respectively. 
function [precision,recall,F1]=prec_recall_F1(y,y_targ,threshold)
epsilon=1e-5;
y=y>threshold;
correct=and((y==y_targ),(y_targ==1));
tot_correct=sum(correct);
tot_pos_predict=sum(y);
tot_pos_targ=sum(y_targ==1);
precision=tot_correct/(tot_pos_predict+epsilon);
recall=tot_correct/(tot_pos_targ+epsilon);
F1=2*precision*recall/(precision+recall+epsilon);
