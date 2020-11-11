clear all; close all; clc
%% parameter setting
N=3;
M=224;
L=10000;
%% algorithm
load data10
X= reshape(X3D,L,M)'; % 3D to 2D
% [~, ~,time_t] = HyperCSI(X,N); % you only need to revise this black box


[A_est, S_est, time] = HyperCSI(X,N);

figure;
map1_est= reshape(S_est(1,:),100,100);
subplot(2,3,1);
imshow(map1_est);title('map 1 est');

map2_est= reshape(S_est(2,:),100,100);
subplot(2,3,2);
imshow(map2_est);title('map 2 est');

map3_est= reshape(S_est(3,:),100,100);
subplot(2,3,3);
imshow(map3_est);title('map 3 est');

subplot(2,3,4);
plot(A_est(:,1)); title('est signature 1');
axis([1 224 0 1]);

subplot(2,3,5);
plot(A_est(:,2)); title('est signature 2');
axis([1 224 0 1]);

subplot(2,3,6);
plot(A_est(:,3)); title('est signature 3');
axis([1 224 0 1]);


% 
