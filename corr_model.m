clear;
clc
input_data=readmatrix("Cleaned_IN_Table.csv");
output_data=readmatrix("Cleaned_OUT_Table.csv");
data=[input_data,output_data];
cor_matrix=zeros(4,1);
cor_matrix2=zeros(4,1);

for i=1:4
    cor_matrix(i,:)=corr(output_data(:,1),input_data(:,i));
end
cor_matrix
%%selecting best 4 correlations
input_best=[input_data(:,1),input_data(:,3)];
train_index=randperm(length(output_data),round(0.7*length(output_data)));
valid_index=setdiff(1:length(output_data),train_index);
input_train=input_best(train_index,:);
output_train=output_data(train_index,:);
input_valid=input_best(valid_index,:);
output_valid=output_data(valid_index,:);
%%%%%%%%%%Regression for two best Variables using corr%%%%%%%%%%%%%%%%%%%
u_data01=[ones(length(output_train),1),input_train(:,1),input_train(:,2)];
beta01=regress(output_train(:,1),u_data01);
u_data02=[ones(length(output_train),1),input_train(:,1),input_train(:,1).^2,input_train(:,2),input_train(:,2).^2];
beta02=regress(output_train(:,1),u_data02);
y_predict1=[ones(length(output_valid),1),input_valid]*beta01;
y_predict2=[ones(length(output_valid),1),input_valid,input_valid.^2]*beta02;
SSE1=sum((y_predict1-output_valid).^2)
SSE2=sum((y_predict2-output_valid).^2)
figure;
plot(1:length(output_valid),y_predict1,"color","r")
hold on
plot(1:length(output_valid),(output_valid(:,1)),"Color","k")
plot(1:length(output_valid),y_predict2,"Color","b")
xlabel("samples");
ylabel("penicillin concentration")
legend("prediction using 1 degree ","original data","prediction using 2 nd degree","Location","northeast")
title("Prediction Using 2 variable for output variable 1")
hold off
for i=1:4
    cor_matrix2(i,:)=corr(output_data(:,2),input_data(:,i));
end
cor_matrix2
u_data001=[ones(length(output_train),1),input_train(:,1),input_train(:,2)];
beta001=regress(output_train(:,1),u_data001);
u_data002=[ones(length(output_train),1),input_train(:,1),input_train(:,1).^2,input_train(:,2),input_train(:,2).^2];
beta002=regress(output_train(:,1),u_data002);
y_predict01=[ones(length(output_valid),1),input_valid]*beta001;
y_predict02=[ones(length(output_valid),1),input_valid,input_valid.^2]*beta002;
SSE1=sum((y_predict01-output_valid).^2)
SSE2=sum((y_predict02-output_valid).^2)
figure;
plot(1:length(output_valid),y_predict01,"color","r")
hold on
plot(1:length(output_valid),(output_valid(:,1)),"Color","k")
plot(1:length(output_valid),y_predict02,"Color","b")
xlabel("samples");
ylabel("penicillin concentration")
legend("prediction using 1 degree ","original data","prediction using 2 nd degree","Location","northeast")
title("Prediction Using 2 variable for output variable 2")
