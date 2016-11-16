function [tscore,dscore,sscore,dtscore,LoRcosttest,NNcosttest] = ...
    MasterCost(X,y,XProcessed,yProcessed,m,v,imbalance,idset,numalg, ...
    NNlambda, LoR2lambda)
%%
% MasterCost computes the machine learning parameters for 5 algorithms (logistic regression, neural networks, support-vector machines, decision trees, and random forest) from a training set of images and tests the parameters on a test set in order to evaluate sensitivity, specificity, and overall accuray. 
% Input arguments include:
% X: a 3-dimensional matrix of features, where each row represents a unique image, each column contains the values for a unique image analysis feature, and each layer represents one of the 3 RGB color layers
% y: a vector containing the pathological diagnoses (ground truth) of the images
% XProcessed: a 2-dimensional matrix of features containing the statistically significant biomarkers (p-value<0.05) from the matrix X
% yProcessed: contains the exact same information as the vector y (only renamed)
% m: the number of images
% v: a vector containing the consecutive whole numbers from 1:m
% imbalance: set to 1 if the ratio of melanoma to nevi images is equal to 1; set to 0 if the aforementioned ratio is not equal to 1
% idset: a vector containing the ID numbers of the images used as a subset if imbalance = 0
% numalg: number of algorithms used in the model (5)
% NNlambda: the value of the regularization parameter for neural networks
% LoR2lambda: the value of the regularization parameter for logistic regression

% Define train and test sets
idtrain = randsample(m, round(m.*0.7));
imtrain = X(idtrain,:);
truetrain = y(idtrain);
idtest = setxor(idtrain,v);
imtest = X(idtest,:);
truetest = y(idtest);
if imbalance
	idtest = idset(idtest);
	idtrain2 = idset(idtrain);
	id = ones(size(XProcessed,1),1);
	id(idtrain2) = 0;
	iddtest2 = find(id == 1);
	dtest2 = XProcessed(iddtest2,:); 
	y2 = yProcessed(iddtest2);
	dtscore = NaN(size(XProcessed,1),numalg);
else
	sscore = []; % filler for balanced datasets
	dtscore = []; % filler for balanced datasets
end

tscore = NaN(size(XProcessed,1),numalg);

%%
% Logistic Regression
initial_theta = zeros(size(imtrain, 2)+1, 1);
% Vary regularization parameter
% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);
% Optimize
[theta, costtrain] = ...
	fminunc(@(t)(costFunctionReg(t, imtrain, truetrain, LoR2lambda)), initial_theta, options);
% Predict, Accuracy
temp = [ones(size(imtrain,1),1),imtrain];
predtrain = sigmoid(temp*theta)>=0.5;
[LoRcosttest, gradtest] = costFunctionReg(theta, imtest, truetest, LoR2lambda);
imtest2 = [ones(size(imtest,1),1),imtest];
LoRpredtest = sigmoid(imtest2*theta)>=0.5;
tscore(idtest,1) = sigmoid(imtest2*theta);
XProcessed2 = [ones(size(XProcessed,1),1),XProcessed];
dscore(:,1) = sigmoid(XProcessed2*theta);
if imbalance
    X2 = [ones(size(X,1),1),X];
    sscore(:,1) = sigmoid(X2*theta);
    X3 = [ones(size(dtest2,1),1),dtest2];
    dtscore(iddtest2,1) = sigmoid(X3*theta);
end
%%
% Neural Networks
input_layer_size = size(imtrain,2);
hidden_layer_size = ceil(size(imtrain,2)./2); %40;
num_labels = 2;
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
% checkNNGradients(3);
options = optimset('MaxIter', 300);
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, imtrain, truetrain, NNlambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
[predtrain, NNcosttrain] = predictnn(Theta1, Theta2, imtrain,truetrain,NNlambda);
[NNpredtest,NNcosttest] = predictnn(Theta1, Theta2, imtest,truetest,NNlambda);
tscore(idtest,2) = scorenn(Theta1, Theta2, imtest);
dscore(:,2) = scorenn(Theta1, Theta2, XProcessed);
if imbalance
    sscore(:,2) = scorenn(Theta1, Theta2, X);
    dtscore(iddtest2,2) = scorenn(Theta1, Theta2, dtest2);
end
%%
% SVM
mdlSVM = fitcsvm(imtrain,truetrain);
mdlSVM = fitPosterior(mdlSVM);
predtrain = predict(mdlSVM, imtrain);
[SVMpredtest,score] = predict(mdlSVM, imtest);
tscore(idtest,3) = score(:,2);
[~,score] = predict(mdlSVM, XProcessed);
dscore(:,3) = score(:,2);
if imbalance
    [~,score] = predict(mdlSVM, X);
    sscore(:,3) = score(:,2);
    [~,score] = predict(mdlSVM, dtest2);
    dtscore(iddtest2,3) = score(:,2); 
end
%%
% Decision Tree
tree = fitctree(imtrain, truetrain,'Cost',[0,1;2,0]);
predtrain = predict(tree, imtrain);
[DTpredtest,score] = predict(tree, imtest);
tscore(idtest,4) = score(:,2);
[~,score] = predict(tree, XProcessed);
dscore(:,4) = score(:,2);
if imbalance
    [~,score] = predict(tree, X);
    sscore(:,4) = score(:,2);
    [~,score] = predict(tree, dtest2);
    dtscore(iddtest2,4) = score(:,2);
end
%%
% Random Forest
B = TreeBagger(10,imtrain, truetrain,'Cost',[0,1;2,0]);
[label,score,cost] = predict(B, imtrain);
[M, I] = max(score, [], 2);
I(I == 1) = 0;
I(I == 2) = 1;
[label,score,cost] = predict(B, imtest);
[M, I] = max(score, [], 2);
I(I == 1) = 0;
I(I == 2) = 1;
RFpredtest = I;
tscore(idtest,5) = score(:,2);
[~,score] = predict(B, XProcessed);
dscore(:,5) = score(:,2);
if imbalance
    [~,score] = predict(B, X);
    sscore(:,5) = score(:,2);
    [~,score] = predict(B, dtest2);
    dtscore(iddtest2,5) = score(:,2);
end
%%
end
