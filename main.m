% main utilizes parallel computing in Matlab in order to run 1000 iterations of MasterCost on a dataset of images.

clear; clc
datanum = 3;
ftsel = 1;
savefig = 0;
%%
name = 'Marghoob';
numiter = 1000;
numalg = 5;
LoR2lambda = 1;
NNlambda = 1;

%%
cd('BiomarkerData')
switch datanum
    case 1
        load('MarghoobXExcel');
        load('MarghoobyExcel');
        imbalance = 0;
    case 2
        load('MarghoobXfixedfiterf2sigmoidnoNaN');
        load('MarghoobyJoe_All');
        imbalance = 0;
    case 3
        load('MarghoobNewMIBs');
        load('MarghoobyJoe_All');
        imbalance = 0;
end
cd ..

% Take most significant biomarker value from the 3 layers for each
% biomarker, p value must be <.05
switch ftsel
    case 1 % t test
        [M2,I,ptable,XProcessed,yProcessed] = pvalue(Matrix_Out, groundtruthworks3);
    case 2 % PCA
        a = squeeze(Matrix_Out(1,:,:));
        b = squeeze(Matrix_Out(2,:,:));
        c = squeeze(Matrix_Out(3,:,:));
        Matrix_Outcat = cat(2,a,b,c);
        [coeff,score,latent,tsquared,explained,mu] = pca(Matrix_Outcat,'NumComponents',30);
        [XProcessed,var2] = pcaextractf(coeff,Matrix_Outcat);
        yProcessed = groundtruthworks3;
end
%%
% Detects NaN
[~, temp1] = find(isnan(XProcessed));
if ~isempty(temp1)
    fprintf('\nWarning: NaN values will affect algorithms.\n');
end
%%
% Load, define X, y
X = XProcessed;
y = yProcessed;
idset = []; % filler for balanced datasets
if imbalance
    mel = find(y == 1);
    lim = length(mel);
    nev = find(y == 0);
    temp = randsample(length(nev),lim);
    nevid = nev(temp);
    idset = cat(1,mel,nevid);
    X = X(idset,:);
    y = y(idset,:);
end
X = zscore(X); % standardize data
XProcessed = zscore(XProcessed);
m = length(y);
v = [1:m]'; % NEED bracket to transpose array
%%
[tscore,dscore,sscore,dtscore] = ...
    MasterCost(X,y,XProcessed,yProcessed,m,v,imbalance,idset,numalg, ...
    NNlambda, LoR2lambda); 
% run first time just to get variables
ttscore = zeros(size(tscore,1),size(tscore,2),numiter);
tdscore = zeros(size(dscore,1),size(dscore,2),numiter);
if imbalance
    tsscore = zeros(size(sscore,1),size(sscore,2),numiter);
    tdtscore = zeros(size(dtscore,1),size(dtscore,2),numiter);
end
%%
matdir = dir('*.mat');
mdir = dir('*.m');
p = gcp;
addAttachedFiles(p,{matdir.name,mdir.name});
% addAttachedFiles(p,{'Master','accuracy','checkNNGradients','costFunctionReg', ...
%     'costFunctionROCarea','LinearRegression','LinearRegressionTest', 'nnCostFunction', ...
%     'predictnn','randInitializeWeights','MarghoobXExcel.mat','MarghoobyExcel.mat', ...
%     'MarghoobXfixedfiterf2sigmoidnoNaN.mat','MarghoobyJoe_All.mat'});
parfor iter = 1:numiter
    [tscore,dscore,sscore,dtscore,LoRcosttest,NNcosttest] = ...
        MasterCost(X,y,XProcessed,yProcessed,m,v,imbalance,idset,numalg, ...
            NNlambda, LoR2lambda); 
    ttscore(:,:,iter) = tscore;
    tdscore(:,:,iter) = dscore;
    tLoRcosttest(:,:,iter) = LoRcosttest;
    tNNcosttest(:,:,iter) = NNcosttest;
    if imbalance
        tsscore(:,:,iter) = sscore;
        tdtscore(:,:,iter) = dtscore;
    end
end
ttscore = nanmean(ttscore,3);
tdscore = mean(tdscore,3);
tLoRcosttest = nanmean(tLoRcosttest,3);
tNNcosttest = nanmean(tNNcosttest,3);
if imbalance
    tsscore = mean(tsscore,3);
    tdtscore = nanmean(tdtscore,3);
end
%%
figure
stats = zeros(numalg+1,4);
for i_alg = 1:size(ttscore,2)  % plots and saves ROC curve
    [Xalg,Yalg,T,AUC] = perfcurve(yProcessed,ttscore(:,i_alg),'1');
    [sp,T] = getspwse98(Xalg,Yalg,T);
    stats(i_alg,:) = [i_alg,sp,T,AUC];
    plot(Xalg,Yalg)
    hold on
end
mttscore = mean(ttscore,2);
[Xalg,Yalg, T, AUC] = perfcurve(yProcessed,mttscore,'1');
plot(Xalg,Yalg,'k-','LineWidth',5)
legend('LoR','NN','SVM','DT','RF','All');
xlabel('1-Specificity')
ylabel('Sensitivity')
title(name)
if savefig
    saveas(gcf, name, 'jpeg')
end
