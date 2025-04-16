clear; close all;

addpath ./ClusteringMeasure
addpath ./twist
path = './data/';

load ./data/NGs.mat;
name = 'NGs';
percentDel = 0.1;  %% missing data rate
Datafold= [path,'Index_',name,'_percentDel_',num2str(percentDel),'.mat'];
load(Datafold)

gt = Y;
cls_num = numel(unique(gt));
k = length(unique(Y));
c = length(unique(Y));

%% NGs
param.alpha = 8;
result = []; perf = [];
    Xc = X;
    ind = Index{1};
    for i=1:length(Xc)   %% set the missing features to 0
        Xci = Xc{i};
        indi = ind(:,i);
        pos = find(indi==0);
        Xci(:,pos)=0; 
        Xc{i} = Xci;
    end

tic   
[Z,A,H,G,E,S,converge] = ASCR(Xc, Y, ind, k, c, param);  %% ASCR algorithm
Z = NormalizeData(Z);
for kk = 1:10
[Clus] = litekmeans(Z, cls_num,'MaxIter',100, 'Replicates',10);
if kk==1
    toc
end

[~,NMI,~] = compute_nmi(Clus, gt);
ACC = Accuracy(Clus, gt);
[f, ~,~]=compute_f(gt, Clus);
[ARI,~,~]=RandIndex(gt, Clus);
[~,~, PUR] = purity(gt,Clus);
pp = [ACC NMI PUR ARI f];
perf = [perf; pp];
end
mean(perf)
std(perf)



