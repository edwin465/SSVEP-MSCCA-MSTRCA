close all
clear all
addpath('..\mytoolbox');

dataset_no=2;
sel_index=[2:7];
if dataset_no==1
    load save_all_sub_acc_th.mat
    num_of_sub=35;num_of_method=length(sel_index);num_of_tw=8;
elseif dataset_no==2
    load save_all_sub_acc_beta.mat
    num_of_sub=70;num_of_method=length(sel_index);num_of_tw=8;
else
end

data=[];
for m=1:num_of_method
    A = squeeze(all_sub_acc(:,:,sel_index(m)));
    data=[data A'];
end

[M,N]=size(data);   
color_rgb=[0 0 0;   
     1 0 0;  
     0 0 1;  
     0 204/255 0;  
     153/255 153/255 0;  
     1 0 1;  
     1 128/255 0;  
     0 128/255 1;  
     0.9 0 0;  
     1 102/255 1;  
     0 1 0];   
title_str='Accuracy comparison';  
x_lab_str='Tw (s)';  
y_lab_str='Accuracy (%)';  
legend_str={'eCCA','ms-eCCA','eTRCA','ms-eTRCA','ms-eCCA+ms-eTRCA','TDCA'};  
x_tick_str={'0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'};  

figure(1);
% fun_barchart_errorbar_plot.m (https://github.com/edwin465/matlab-mytoolbox-plot)
fun_barchart_errorbar_plot(data,num_of_method,num_of_tw,color_rgb,title_str,x_lab_str,y_lab_str,legend_str(1:num_of_method),x_tick_str);

dat_x=[0.3:0.1:1.0];
dat_y1=squeeze(all_sub_acc(:,:,6));
dat_y1=dat_y1';
dat_y2=squeeze(all_sub_acc(:,:,7));
dat_y2=dat_y2';
title_str='ms-eCCA+ms-eTRCA vs TDCA';
x_lab_str='Tw (s)';
y_lab_str='Accuracy (%)';
legend_str1='ms-eCCA+ms-eTRCA';legend_str2='TDCA';
test_pval0=1; % Calculate the paired t-test result (tail: both))
% fun_errorbarchart.m (https://github.com/edwin465/matlab-mytoolbox-plot)
fun_errorbarchart(dat_x,dat_y1,dat_y2,test_pval0,title_str,x_lab_str,y_lab_str,legend_str1,legend_str2);


