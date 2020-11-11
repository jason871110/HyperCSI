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

d = mean(X,2); %���C��row������
U = X-d;      %�����XU�V�q
cor=U*U';     %corvarience matrix
[V,~] = eig(cor);  %eigen value and vector
C = V(:,M-1:end);  %���X�̤j����Ӽƭ�
dr_data_ori = C'*U;    %dimension-reduced
filter_dr_data=unique(dr_data_ori','rows','stable');  %�L�o��dimension reduced data�����ƪ����
dr_data=filter_dr_data';  %��ܦ^�i�ϥΪ���Ʈ榡->�L�o���
[~,L] = size(dr_data);    %�����L�o������Ƥj�p

% dr_data=gpuArray(dr_data);


%% �i�ϥιL�o��ƶ}�l
%% SPA

% value_li=gpuArray(zeros(3,L));          %�s���Ĥ@����v���Ҧ��I
value_li=zeros(3,L);          %�s���Ĥ@����v���Ҧ��I
X_set = [dr_data; ones(1,L)]; %�즳��DR�[�J1�o��row
[dist_1 ,index_1] = max(sum( X_set.^2 ));  %%��Ĥ@���I���Z���Mindex
normal_param=X_set(:,index_1);             %1���k�V�q
for i=1:L
    current_param=X_set(:,i);            %loop�쪺�I
    ccc=dot(normal_param,current_param');   %���n
    value=current_param-ccc/dist_1*normal_param;  %��v����
    value_li(:,i)=value;                          %��v���
end


[dist_2, index_2] = max(sum(value_li.^2 ));       %�ĤG���I
normal_param_2=value_li(:,index_2);              %2���k�V�q

[~,index_3]=max(sum((value_li-value_li(:,index_2)).^2));   %�ֳtSPA
% for i=1:L
%     current_param=value_li(:,i);                %loop�쪺�I
%     ccc=dot(normal_param_2,current_param');     %���n
%     value=current_param-ccc/dist_2*normal_param_2;%��v����
%     value_li(:,i)=value;                          %��v���
% end
% [~, index_3] = max(sum(value_li.^2 ));       %�ĤT���I


%% Step3
spot_1=dr_data(:,index_1);   %���I1
spot_2=dr_data(:,index_2);   %���I2
spot_3=dr_data(:,index_3);   %���I3
lenght_li=[sum((spot_1-spot_2).^2),sum((spot_1-spot_3).^2),sum((spot_2-spot_3).^2)];  %%3���
min_length=min(lenght_li)/2;  %%�̵u��@�b
%% �D�k�V�q
% �Db_1
[b_1_init]=get_normal_vector_v2(spot_3,spot_2);
% �Db_2
% [b_2_init] = get_normal_vector(spot_2,spot_3,spot_1);
[b_2_init]=get_normal_vector_v2(spot_2,spot_1);
% �Db_3
% [b_3_init] = get_normal_vector(spot_1,spot_2,spot_3);
[b_3_init]=get_normal_vector_v2(spot_1,spot_3);

%%a3-a2-�Ĥ@����
[index_3_] =get_max_dot_value_index(L,dr_data,min_length,index_3,b_1_init);
[index_2_] =get_max_dot_value_index(L,dr_data,min_length,index_2,b_1_init);
[b_1_final]=get_normal_vector_v2(dr_data(:,index_3_),dr_data(:,index_2_));

%%a2-a1-�ĤG����
[index_2_] =get_max_dot_value_index(L,dr_data,min_length,index_2,b_2_init);
[index_1_] =get_max_dot_value_index(L,dr_data,min_length,index_1,b_2_init);
[b_2_final]=get_normal_vector_v2(dr_data(:,index_2_),dr_data(:,index_1_));

%%a1-a3-�ĤT����
[index_1_] =get_max_dot_value_index(L,dr_data,min_length,index_1,b_3_init);
[index_3_] =get_max_dot_value_index(L,dr_data,min_length,index_3,b_3_init);
[b_3_final]=get_normal_vector_v2(dr_data(:,index_1_),dr_data(:,index_3_));


%% Step4.
%��X���n�̤j��
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
    
%% Step5.��Xalpha

b_no_1=[b_2_final,b_3_final]';
h_no_1=[max_b2,max_b3]';
alpha_no_1=b_no_1\h_no_1; 

b_no_2=[b_1_final,b_3_final]';
h_no_2=[max_b1,max_b3]';
alpha_no_2=b_no_2\h_no_2;

b_no_3=[b_1_final,b_2_final]';
h_no_3=[max_b1,max_b2]';
alpha_no_3=b_no_3\h_no_3;

%% �f�VPCA�^�즳������
vertex_1=C*alpha_no_1+d;
vertex_2=C*alpha_no_2+d;
vertex_3=C*alpha_no_3+d;
A_est=[vertex_1,vertex_2,vertex_3];


%% �i�ϥιL�o��Ƶ���
%% Step7.�p�⪫����
dr_data=dr_data_ori;  %��^��Ӫ�dimension reduced data
portion_1=(max_b1-b_1_final'*dr_data)/(max_b1-b_1_final'*alpha_no_1);
portion_2=(max_b2-b_2_final'*dr_data)/(max_b2-b_2_final'*alpha_no_2);
portion_3=(max_b3-b_3_final'*dr_data)/(max_b3-b_3_final'*alpha_no_3);
S_est=[portion_1;portion_2;portion_3];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
time = etime(clock,t0);

%% subprogram 1:�D�k�V�q_1
function [normal_vec] = get_normal_vector(spot_start,spot_proj,spot_end)
vec_start_proj = spot_proj-spot_start;   %�n��v�����q
vec_start_end=spot_end-spot_start;     %��v��V
dist_vec_start_end = sum(vec_start_end.^2); %��v���q�Z��
dot_proj_end=dot(vec_start_proj,vec_start_end'); %��v�һݤ��n��
spot_start_proj=dot_proj_end/dist_vec_start_end*vec_start_end; %��v����
normal_vec = spot_start_proj-vec_start_proj; %�k�V�q
return;
% end
%% subprogram 2:�D�k�V�q_2
function [normal_vec] = get_normal_vector_v2(spot_start,spot_end)
vec_start_end=spot_end-spot_start;     %��v��V
normal_vec=[vec_start_end(2,1);-1*vec_start_end(1,1)]; %�G���k�V�q�ޥ�
return;
%% subprogram  3:��ꤺ���n�̤j��
function [index] =get_max_dot_value_index(L,dr_data,min_length,spot_index,normal_vector)
max_dot_value=0; %%��̤j���n�I����
max_dot_index=1; %%��̤j���n�I��index
%a3-�Ĥ@����-�Ĥ@�I
for i=1:L
    if (sum((dr_data(:,spot_index)-dr_data(:,i)).^2)<min_length) %�T�{�O�_�b��νd��
        temp_dot_value=dot(dr_data(:,i),normal_vector);            %���n��
        if ( temp_dot_value>max_dot_value)
             max_dot_value=temp_dot_value;
             max_dot_index=i;
        end
    end
end
index = max_dot_index;

