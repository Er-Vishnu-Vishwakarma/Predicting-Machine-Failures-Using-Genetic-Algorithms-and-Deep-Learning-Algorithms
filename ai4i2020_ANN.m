%% ai4i2020_ANN.m
% MATLAB Pipeline: GA + ANN for Predictive Maintenance
% - Includes: Train/Test split, Normalization, Correlation/VIF pruning,
%   SMOTE (or fallback oversampling), GA-based hyperparameter/feature search,
%   ANN training, metrics (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)

%% CONFIG
csvPath   = '/MATLAB Drive/ai4i2020.csv.csv'; % uploaded dataset
TARGET    = 'MachineFailure';  % expected target column name (after renaming)
ID_COLS   = {'UDI'};
FAILURE_PARTS = {'TWF','HDF','PWF','OSF','RNF'};
FEATURES_FORCE_KEEP = {}; 
VIF_MAX      = 10; 
K_FOLDS      = 5;  
USE_GA       = true; 
SEED         = 42;  

% GA search spaces
GA_space.hidden1    = [4  64];
GA_space.hidden2    = [0  64];
GA_space.epochs     = [50 200];
GA_space.lr_scaled  = [1  50];   % -> /1000 = 0.001..0.050
GA_space.l2_scaled  = [0 100];   % -> /1000 = 0..0.100
GA_space.useFeatSel = [0  1];

%% 1) Load data
rngdefault(SEED);
T = readtable(csvPath);

% fix column names
T.Properties.VariableNames = matlab.lang.makeValidName(T.Properties.VariableNames);

% drop ID & categorical/text columns
for c = 1:numel(ID_COLS)
    if ismember(ID_COLS{c}, T.Properties.VariableNames)
        T.(ID_COLS{c}) = [];
    end
end
for c = 1:numel(FAILURE_PARTS)
    if ismember(FAILURE_PARTS{c}, T.Properties.VariableNames)
        T.(FAILURE_PARTS{c}) = [];
    end
end

% automatically drop non-numeric predictors
vars = T.Properties.VariableNames;
isNum = varfun(@isnumeric, T, 'OutputFormat','uniform');
for i = 1:numel(vars)
    if ~isNum(i) && ~strcmp(vars{i}, TARGET)
        fprintf('Dropping non-numeric variable: %s\n', vars{i});
        T.(vars{i}) = [];
    end
end

% target variable
y = T.(TARGET);
if ~islogical(y)
    if isnumeric(y), y = y~=0; else, y = strcmp(string(y),'1'); end
end
T.(TARGET) = [];
X = T;

%% 2) Train/Test split
cvHoldout = cvpartition(y,'Holdout',0.2);
Xtrain = X(training(cvHoldout),:); ytrain = y(training(cvHoldout),:);
Xtest  = X(test(cvHoldout),:);    ytest  = y(test(cvHoldout),:);

%% 3) Normalization
[XtrainN,scaler] = minmax_fit_transform(Xtrain);
XtestN = minmax_apply(Xtest, scaler);

%% 4) Correlation pruning
[XtrainCorrPruned, keepCorr] = corr_prune(XtrainN, FEATURES_FORCE_KEEP);
XtestCorrPruned = XtestN(:, keepCorr);

%% 5) VIF pruning
[XtrainPruned, keepVIF] = vif_prune(XtrainCorrPruned, VIF_MAX);
XtestPruned = XtestCorrPruned(:, keepVIF);

%% 6) SMOTE (fallback: oversampling)
try
    [XtrainBal, ytrainBal] = smote_binary(XtrainPruned, ytrain, 5, 1);
catch
    [XtrainBal, ytrainBal] = simple_oversample(XtrainPruned, ytrain);
end

%% 7) GA + ANN training
Xtr = table2array(XtrainBal); Xte = table2array(XtestPruned);
cvp = cvpartition(ytrainBal,'KFold',K_FOLDS);

if USE_GA
    nFeat = size(Xtr,2);
    LB = [GA_space.hidden1(1), GA_space.hidden2(1), GA_space.epochs(1), GA_space.lr_scaled(1), GA_space.l2_scaled(1), zeros(1,nFeat), GA_space.useFeatSel(1)];
    UB = [GA_space.hidden1(2), GA_space.hidden2(2), GA_space.epochs(2), GA_space.lr_scaled(2), GA_space.l2_scaled(2), ones(1,nFeat),  GA_space.useFeatSel(2)];
    nvars = numel(LB); intcon = 1:nvars;
    opts = optimoptions('ga','Display','iter','MaxGenerations',10,'PopulationSize',20,'UseParallel',false);
    fitnessFcn = @(z) -cv_auc_from_vec(z, Xtr, ytrainBal, cvp);
    [zbest, fval] = ga(fitnessFcn, nvars, [],[],[],[], LB, UB, [], intcon, opts);
    [hp, featMask] = decode_vec(zbest, size(Xtr,2));
    if hp.useFeatSel, Xtr_sel = Xtr(:, featMask); Xte_sel = Xte(:, featMask);
    else, Xtr_sel = Xtr; Xte_sel = Xte; end
    net = train_ann(Xtr_sel, ytrainBal, hp);
    [teScores, tePred] = predict_ann(net, Xte_sel);
else
    hp.h1=32; hp.h2=16; hp.epochs=100; hp.lr=0.01; hp.l2=0.001;
    net = train_ann(Xtr, ytrainBal, hp);
    [teScores, tePred] = predict_ann(net, Xte);
end

[metrics, ~] = classification_metrics(ytest, tePred, teScores);
disp(metrics);

figure; confusionchart(ytest, tePred,'Title','Confusion Matrix (Test)');
[Xroc,Yroc,~,AUC] = perfcurve(ytest, teScores, true);
figure; plot(Xroc,Yroc,'LineWidth',2); grid on;
xlabel('FPR'); ylabel('TPR'); title(sprintf('ROC (AUC=%.4f)',AUC));

%% === Helper Functions ===
function rngdefault(seed), if exist('rng','file'), rng(seed); else, rand('seed',seed); randn('seed',seed); end, end
function [Xn,sc] = minmax_fit_transform(Xtbl)
    Xmat = table2array(varfun(@double,Xtbl)); xmin = min(Xmat,[],1); xmax = max(Xmat,[],1);
    span = xmax - xmin; span(span==0) = 1;
    Xn = array2table((Xmat - xmin)./span,'VariableNames',Xtbl.Properties.VariableNames);
    sc.xmin = xmin; sc.span = span; sc.names = Xtbl.Properties.VariableNames;
end
function Xn = minmax_apply(Xtbl, sc)
    Xmat = table2array(varfun(@double,Xtbl));
    [~,idx] = ismember(sc.names, Xtbl.Properties.VariableNames);
    Xn = array2table((Xmat(:,idx)-sc.xmin)./sc.span,'VariableNames',sc.names);
end
function [Xout, keepIdx] = corr_prune(Xtbl, forceKeep)
    Xmat = table2array(Xtbl); cn = Xtbl.Properties.VariableNames;
    C = corr(Xmat,'Rows','pairwise'); thr = 0.9; remove = false(1,size(C,1));
    for i=1:size(C,1)
        for j=i+1:size(C,1)
            if abs(C(i,j))>thr
                if ismember(cn{j}, forceKeep), remove(i)=true; else, remove(j)=true; end
            end
        end
    end
    keepIdx = find(~remove); Xout = Xtbl(:,keepIdx);
end
function [Xout, keepIdx] = vif_prune(Xtbl, vmax)
    X = table2array(Xtbl); nfeat=size(X,2); keep=true(1,nfeat); changed=true;
    while changed
        changed=false; V=nan(1,nfeat);
        for i=1:nfeat
            if ~keep(i), continue; end
            idx=find(keep); idx(idx==i)=[]; if isempty(idx), V(i)=1; continue; end
            Xi=X(:,idx); yi=X(:,i); b=regress(yi,[ones(size(Xi,1),1) Xi]);
            yhat=[ones(size(Xi,1),1) Xi]*b;
            R2=1-sum((yi-yhat).^2)/sum((yi-mean(yi)).^2);
            V(i)=1/(1-R2+eps);
        end
        if any(V>vmax), [~,rm]=max(V); keep(rm)=false; changed=true; end
    end
    keepIdx=find(keep); Xout=Xtbl(:,keepIdx);
end
function [Xb,yb] = smote_binary(Xtbl,y,k,over)
    X=table2array(Xtbl); cls0=find(~y); cls1=find(y);
    if numel(cls1)<numel(cls0), minIdx=cls1; minLab=true; else, minIdx=cls0; minLab=false; end
    majN=max(numel(cls0),numel(cls1)); minN=numel(minIdx);
    synthNeeded=majN-minN; if synthNeeded<=0, Xb=Xtbl; yb=y; return; end
    mdl=createns(X(minIdx,:),'NSMethod','kdtree'); S=zeros(synthNeeded,size(X,2));
    for s=1:synthNeeded
        i=randi(minN); [idx,~]=knnsearch(mdl,X(minIdx(i),:),'K',min(k+1,minN));
        j=idx(randi(numel(idx)-1)+1); lam=rand();
        S(s,:)=X(minIdx(i),:)+lam*(X(minIdx(j),:)-X(minIdx(i),:));
    end
    Xnew=[X;S]; ynew=[y; repmat(minLab,synthNeeded,1)];
    Xb=array2table(Xnew,'VariableNames',Xtbl.Properties.VariableNames); yb=ynew;
end
function [Xb,yb] = simple_oversample(Xtbl,y)
    X=table2array(Xtbl); cls0=find(~y); cls1=find(y);
    if numel(cls1)<numel(cls0), reps=randsample(cls1,numel(cls0)-numel(cls1),true); addLab=true;
    else, reps=randsample(cls0,numel(cls1)-numel(cls0),true); addLab=false; end
    Xnew=[X;X(reps,:)]; ynew=[y; repmat(addLab,numel(reps),1)];
    Xb=array2table(Xnew,'VariableNames',Xtbl.Properties.VariableNames); yb=ynew;
end
function [hp,mask] = decode_vec(z,nFeat)
    hp.h1=round(z(1)); hp.h2=round(z(2)); hp.epochs=round(z(3));
    hp.lr=round(z(4))/1000; hp.l2=round(z(5))/1000;
    maskBits=round(z(6:5+nFeat)); hp.useFeatSel=logical(round(z(5+nFeat+1)));
    if ~hp.useFeatSel, mask=true(1,nFeat);
    else, if sum(maskBits)==0, maskBits(randi(nFeat))=1; end, mask=logical(maskBits); end
end
function f = cv_auc_from_vec(z,X,y,cvp)
    [hp,mask]=decode_vec(z,size(X,2)); Xs=X(:,mask); aucs=zeros(cvp.NumTestSets,1);
    for k=1:cvp.NumTestSets
        idxTr=training(cvp,k); idxTe=test(cvp,k);
        net=train_ann(Xs(idxTr,:),y(idxTr),hp); [scores,~]=predict_ann(net,Xs(idxTe,:));
        [~,~,~,au]=perfcurve(y(idxTe),scores,true); aucs(k)=au;
    end, f=mean(aucs);
end
function net = train_ann(X,y,hp)
    X=X'; t=double(y(:))';
    if hp.h2>0, net=patternnet([hp.h1 hp.h2],'trainscg');
    else, net=patternnet(hp.h1,'trainscg'); end
    net.performParam.regularization=min(max(hp.l2,0),0.5);
    net.trainParam.lr=hp.lr; net.trainParam.epochs=hp.epochs; net.trainParam.showWindow=false;
    net.divideFcn='dividerand'; net.divideParam.trainRatio=0.8; net.divideParam.valRatio=0.2; net.divideParam.testRatio=0.0;
    net=train(net,X,t);
end
function [scores,yhat] = predict_ann(net,X), X=X'; p=net(X); scores=p(:); yhat=scores>=0.5; end
function [M,C] = classification_metrics(ytrue,ypred,scores)
    ytrue=logical(ytrue(:)); ypred=logical(ypred(:));
    TP=sum(ypred & ytrue); TN=sum(~ypred & ~ytrue); FP=sum(ypred & ~ytrue); FN=sum(~ypred & ytrue);
    acc=(TP+TN)/numel(ytrue); prec=TP/max(1,TP+FP); rec=TP/max(1,TP+FN); f1=2*prec*rec/max(1e-12,prec+rec);
    [~,~,~,auc]=perfcurve(ytrue,scores,true);
    M=table(acc,prec,rec,f1,auc,TP,FP,TN,FN,'VariableNames',{'Accuracy','Precision','Recall','F1','ROC_AUC','TP','FP','TN','FN'});
    C=[TP FP; FN TN];
end











