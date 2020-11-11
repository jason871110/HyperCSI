%======================================================================
%  Input
%  X is M-by-L data matrix, where M is the number of spectral bands and L is the number of pixels.
%  N is the number of endmembers.
%----------------------------------------------------------------------
%  Output
%  A_est is M-by-N mixing matrix whose columns are estimated endmember signatures.
%  S_est is N-by-L source matrix whose rows are estimated abundance maps.
%  time is the computation time (in secs).
%========================================================================

function [A_est, S_est, time] = HyperCSI_10_v3(X,N)

t0 = clock;
M=224;
L=10000;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PCA

d = mean(X,2); %取每個row的平均
U = X-d;      %平移出U向量
cor=U*U';     %corvarience matrix
[V,~] = eig(cor);  %eigen value and vector
C = V(:,M-1:end);  %取出最大的兩個數值
dr_data_ori = C'*U;    %dimension-reduced
filter_dr_data=unique(dr_data_ori','rows','stable');  %過濾掉dimension reduced data內重複的資料
dr_data=filter_dr_data';  %轉至回可使用的資料格式->過濾資料
[~,L] = size(dr_data);    %拿取過濾完的資料大小

% dr_data=gpuArray(dr_data);


%% 可使用過濾資料開始
%% SPA

% value_li=gpuArray(zeros(3,L));          %存取第一次投影的所有點
value_li=zeros(3,L);          %存取第一次投影的所有點
X_set = [dr_data; ones(1,L)]; %原有的DR加入1這項row
[dist_1 ,index_1] = max(sum( X_set.^2 ));  %%找第一頂點的距離和index
normal_param=X_set(:,index_1);             %1號法向量
for i=1:L
    current_param=X_set(:,i);            %loop到的點
    ccc=dot(normal_param,current_param');   %內積
    value=current_param-ccc/dist_1*normal_param;  %投影公式
    value_li(:,i)=value;                          %投影資料
end


[dist_2, index_2] = max(sum(value_li.^2 ));       %第二頂點
normal_param_2=value_li(:,index_2);              %2號法向量

[~,index_3]=max(sum((value_li-value_li(:,index_2)).^2));   %快速SPA
% for i=1:L
%     current_param=value_li(:,i);                %loop到的點
%     ccc=dot(normal_param_2,current_param');     %內積
%     value=current_param-ccc/dist_2*normal_param_2;%投影公式
%     value_li(:,i)=value;                          %投影資料
% end
% [~, index_3] = max(sum(value_li.^2 ));       %第三頂點


%% Step3
spot_1=dr_data(:,index_1);   %頂點1
spot_2=dr_data(:,index_2);   %頂點2
spot_3=dr_data(:,index_3);   %頂點3
lenght_li=[sum((spot_1-spot_2).^2),sum((spot_1-spot_3).^2),sum((spot_2-spot_3).^2)];  %%3邊長
min_length=min(lenght_li)/2;  %%最短邊一半
%% 求法向量
% 求b_1
[b_1_init]=get_normal_vector_v2(spot_3,spot_2);
% 求b_2
% [b_2_init] = get_normal_vector(spot_2,spot_3,spot_1);
[b_2_init]=get_normal_vector_v2(spot_2,spot_1);
% 求b_3
% [b_3_init] = get_normal_vector(spot_1,spot_2,spot_3);
[b_3_init]=get_normal_vector_v2(spot_1,spot_3);

%%a3-a2-第一組邊
[index_3_] =get_max_dot_value_index(L,dr_data,min_length,index_3,b_1_init);
[index_2_] =get_max_dot_value_index(L,dr_data,min_length,index_2,b_1_init);
[b_1_final]=get_normal_vector_v2(dr_data(:,index_3_),dr_data(:,index_2_));

%%a2-a1-第二組邊
[index_2_] =get_max_dot_value_index(L,dr_data,min_length,index_2,b_2_init);
[index_1_] =get_max_dot_value_index(L,dr_data,min_length,index_1,b_2_init);
[b_2_final]=get_normal_vector_v2(dr_data(:,index_2_),dr_data(:,index_1_));

%%a1-a3-第三組邊
[index_1_] =get_max_dot_value_index(L,dr_data,min_length,index_1,b_3_init);
[index_3_] =get_max_dot_value_index(L,dr_data,min_length,index_3,b_3_init);
[b_3_final]=get_normal_vector_v2(dr_data(:,index_1_),dr_data(:,index_3_));


%% Step4.
%找出內積最大值
max_b1=0;
max_b2=0;
max_b3=0;
for i=1:L
    b1_value=dot(dr_data(:,i),b_1_final);
    b2_value=dot(dr_data(:,i),b_2_final);
    b3_value=dot(dr_data(:,i),b_3_final);
    if  b1_value>max_b1
        max_b1= b1_value;
    end
    if  b2_value>max_b2
        max_b2= b2_value;
    end
    if  b3_value>max_b3
        max_b3= b3_value;
    end
end
    
%% Step5.找出alpha

b_no_1=[b_2_final,b_3_final]';
h_no_1=[max_b2,max_b3]';
alpha_no_1=b_no_1\h_no_1; 

b_no_2=[b_1_final,b_3_final]';
h_no_2=[max_b1,max_b3]';
alpha_no_2=b_no_2\h_no_2;

b_no_3=[b_1_final,b_2_final]';
h_no_3=[max_b1,max_b2]';
alpha_no_3=b_no_3\h_no_3;

%% 逆向PCA回原有的維度
vertex_1=C*alpha_no_1+d;
vertex_2=C*alpha_no_2+d;
vertex_3=C*alpha_no_3+d;
A_est=[vertex_1,vertex_2,vertex_3];


%% 可使用過濾資料結束
%% Step7.計算物質比例
dr_data=dr_data_ori;  %改回原來的dimension reduced data
portion_1=(max_b1-b_1_final'*dr_data)/(max_b1-b_1_final'*alpha_no_1);
portion_2=(max_b2-b_2_final'*dr_data)/(max_b2-b_2_final'*alpha_no_2);
portion_3=(max_b3-b_3_final'*dr_data)/(max_b3-b_3_final'*alpha_no_3);
S_est=[portion_1;portion_2;portion_3];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
time = etime(clock,t0);

%% subprogram 1:求法向量_1
function [normal_vec] = get_normal_vector(spot_start,spot_proj,spot_end)
vec_start_proj = spot_proj-spot_start;   %要投影的像量
vec_start_end=spot_end-spot_start;     %投影方向
dist_vec_start_end = sum(vec_start_end.^2); %投影像量距離
dot_proj_end=dot(vec_start_proj,vec_start_end'); %投影所需內積值
spot_start_proj=dot_proj_end/dist_vec_start_end*vec_start_end; %投影公式
normal_vec = spot_start_proj-vec_start_proj; %法向量
return;
% end
%% subprogram 2:求法向量_2
function [normal_vec] = get_normal_vector_v2(spot_start,spot_end)
vec_start_end=spot_end-spot_start;     %投影方向
normal_vec=[vec_start_end(2,1);-1*vec_start_end(1,1)]; %二維法向量技巧
return;
%% subprogram  3:找圓內內積最大值
function [index] =get_max_dot_value_index(L,dr_data,min_length,spot_index,normal_vector)
max_dot_value=0; %%找最大內積點的值
max_dot_index=1; %%找最大內積點的index
%a3-第一組邊-第一點
for i=1:L
    if (sum((dr_data(:,spot_index)-dr_data(:,i)).^2)<min_length) %確認是否在圓形範圍
        temp_dot_value=dot(dr_data(:,i),normal_vector);            %內積值
        if ( temp_dot_value>max_dot_value)
             max_dot_value=temp_dot_value;
             max_dot_index=i;
        end
    end
end
index = max_dot_index;

