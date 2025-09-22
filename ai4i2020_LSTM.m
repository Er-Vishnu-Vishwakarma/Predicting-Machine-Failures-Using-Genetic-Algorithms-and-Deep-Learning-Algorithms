%% Final: LSTM + GA + SMOTE (Corrected Input Dimension for ai4i2020.csv)
clc; clear; close all;

%% --- Load Dataset
T = readtable('ai4i2020.csv.csv');  % Adjust filename if needed

% Target (Machine Failure column)
targetCol = find(strcmpi(T.Properties.VariableNames,'MachineFailure') | ...
                 strcmpi(T.Properties.VariableNames,'machine_failure'),1);
if isempty(targetCol)
    error('Target column MachineFailure not found.');
end
Yraw = T{:,targetCol};
T(:,targetCol) = [];

% Convert predictors to numeric
X = zeros(size(T,1), width(T));
for j = 1:width(T)
    col = T{:,j};
    if isnumeric(col)
        X(:,j) = col;
    else
        X(:,j) = double(categorical(string(col)));
    end
end

% Target to 0/1
if isnumeric(Yraw) && all(ismember(unique(Yraw),[0 1]))
    Y = Yraw;
else
    [~,~,ic] = unique(Yraw);
    Y = ic-1;
end

%% Normalize Features
X = normalize(X);

%% Apply SMOTE
[X_bal,Y_bal] = SMOTE_balanced(X,Y,5);

%% Train-Test Split
cv = cvpartition(length(Y_bal),'HoldOut',0.2);
XTrain = X_bal(training(cv),:);
YTrain = Y_bal(training(cv));
XTest  = X_bal(test(cv),:);
YTest  = Y_bal(test(cv));

%% Convert to cell arrays for LSTM (each sample = [features x 1])
XTrainCell = makeCell(XTrain);
XTestCell  = makeCell(XTest);

%% Define GA fitness function
fitnessFcn = @(p) lstmFitness_safe(p,XTrain,YTrain,XTest,YTest);

% GA options
nvars = 3;
lb = [10 0.001 20];
ub = [200 0.1 100];
opts = optimoptions('ga','PopulationSize',10,'MaxGenerations',5,'Display','iter');

[bestParams,~] = ga(fitnessFcn,nvars,[],[],[],[],lb,ub,[],opts);

hiddenUnits = round(bestParams(1));
learnRate   = bestParams(2);
epochs      = round(bestParams(3));

fprintf('\nGA Best Params -> Hidden=%d, LR=%.4f, Epochs=%d\n',hiddenUnits,learnRate,epochs);

%% Define final LSTM
numFeatures = size(X,2);
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(hiddenUnits,'OutputMode','last')
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

opts = trainingOptions('adam','InitialLearnRate',learnRate,...
    'MaxEpochs',epochs,'MiniBatchSize',64,'Verbose',false);

net = trainNetwork(XTrainCell,categorical(YTrain),layers,opts);

%% Predictions + ROC Curve
YPred = classify(net,XTestCell);
YPredNum = double(YPred)-1;

confMat = confusionmat(YTest,YPredNum);
accuracy = sum(YPredNum==YTest)/numel(YTest);

TP=confMat(2,2); FP=confMat(1,2); FN=confMat(2,1); TN=confMat(1,1);
precision = TP/(TP+FP+eps);
recall    = TP/(TP+FN+eps);
f1        = 2*(precision*recall)/(precision+recall+eps);

probs = predict(net,XTestCell);
if size(probs,2) >= 2
    posScores = probs(:,2);
else
    posScores = YPredNum;
end

% ROC Curve + AUC
[Xroc,Yroc,~,AUC] = perfcurve(YTest,posScores,1);

disp('Confusion Matrix:'); disp(confMat);
fprintf('Accuracy: %.4f\nPrecision: %.4f\nRecall: %.4f\nF1: %.4f\nROC-AUC: %.4f\n',...
    accuracy,precision,recall,f1,AUC);

figure;
plot(Xroc,Yroc,'-b','LineWidth',2); hold on;
plot([0 1],[0 1],'--r'); % diagonal line
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('ROC Curve (AUC = %.4f)',AUC));
grid on;

%% Cross Validation
K=5; cvp=cvpartition(Y_bal,'KFold',K); cvAcc=zeros(K,1);
for k=1:K
    tr=training(cvp,k); te=test(cvp,k);
    Xtr=makeCell(X_bal(tr,:)); Ytr=categorical(Y_bal(tr));
    Xte=makeCell(X_bal(te,:)); Yte=Y_bal(te);
    netCV=trainNetwork(Xtr,Ytr,layers,opts);
    Ycv=classify(netCV,Xte);
    cvAcc(k)=sum(double(Ycv)-1==Yte)/numel(Yte);
end
fprintf('Cross-Validation Mean Accuracy: %.4f\n',mean(cvAcc));

%% ---------------- Helper Functions ----------------
function Xcell = makeCell(X)
    % Each sample must be [numFeatures x 1]
    N=size(X,1); Xcell=cell(1,N);
    for i=1:N
        Xcell{i}=X(i,:)';   % transpose row -> column
    end
end

function err = lstmFitness_safe(p,Xtr,Ytr,Xte,Yte)
    hidden=round(p(1)); lr=p(2); ep=round(p(3));
    numFeat=size(Xtr,2);
    layers=[sequenceInputLayer(numFeat)
            lstmLayer(hidden,'OutputMode','last')
            fullyConnectedLayer(2)
            softmaxLayer
            classificationLayer];
    opts=trainingOptions('adam','InitialLearnRate',lr,'MaxEpochs',ep,'MiniBatchSize',64,'Verbose',false);
    try
        net=trainNetwork(makeCell(Xtr),categorical(Ytr),layers,opts);
        Yp=classify(net,makeCell(Xte));
        acc=sum(double(Yp)-1==Yte)/numel(Yte);
        err=1-acc;
    catch
        err=1;
    end
end

function [X_out,Y_out] = SMOTE_balanced(X_in,Y_in,k)
    classes=unique(Y_in); X_out=X_in; Y_out=Y_in;
    if numel(classes)~=2, return; end
    n0=sum(Y_in==classes(1)); n1=sum(Y_in==classes(2));
    if n0==n1, return; end
    if n0<n1, minC=classes(1); majCount=n1;
    else, minC=classes(2); majCount=n0; end
    X_min=X_in(Y_in==minC,:); n_min=size(X_min,1);
    N=majCount-n_min; synth=zeros(N,size(X_in,2));
    if n_min<=1
        synth=repmat(X_min,ceil(N/n_min),1); synth=synth(1:N,:);
    else
        if k>=n_min, k=n_min-1; end
        neighbors=knnsearch(X_min,X_min,'K',k+1);
        for i=1:N
            idx=randi(n_min);
            neighs=neighbors(idx,2:end);
            nn=X_min(neighs(randi(length(neighs))),:);
            diff=nn-X_min(idx,:); gap=rand(1,size(X_in,2));
            synth(i,:)=X_min(idx,:)+gap.*diff;
        end
    end
    X_out=[X_in; synth]; Y_out=[Y_in; repmat(minC,N,1)];
end


