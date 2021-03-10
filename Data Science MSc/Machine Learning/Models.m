%% Best two Models A & D:

% Clear workspace and Command window
clear all; clc; close all;
% Load data in
load('data.mat');
% Drop categorical from data before we model
data(:,2:2) = [];

%% Model A - Fit a Decision Tree Classification Model to our data:

% Partition the data using hold out method at 70-30
cvpt = cvpartition(data.fraudulent,"HoldOut",0.3);
% Set training dataset
trainData = data(training(cvpt),:);
% Set testing dataset
testData = data(test(cvpt),:);

% Create Tree model using training data
A_mdl = fitctree(trainData,"fraudulent");

% text description
view(A_mdl);
% Graphic description
view(A_mdl,'mode','graph');

% Calculating the resubsitution loss for the training data
A_errTrain = resubLoss(A_mdl);
% Calculate the loss on the test data
A_errTest = loss(A_mdl,testData);
% Display the loss values
disp("Model A - Tree Train Loss: " + A_errTrain);
disp("Model A - Tree Test Loss: " + A_errTest);

% Predict the values on the test data
A_pred = predict(A_mdl,testData);

% Create 10 K-Fold partition on model A which used the training data
A2_mdl = crossval(A_mdl,'kfold', 10);
% Calculate the K-Fold loss
A2_errK = kfoldLoss(A2_mdl);
% Display the K-Fold loss
disp("Model A - Tree K-Fold Loss: " + A2_errK);

% Predict the values on the K-Fold data
A2_pred = kfoldPredict(A2_mdl);

% Calculating FN, FP, TP & TF for model 1
A_Mdl_FN = sum((trainData.fraudulent == 1) & (A2_pred == 0));
A_Mdl_FP = sum((trainData.fraudulent == 0) & (A2_pred == 1));
A_Mdl_TP = sum((trainData.fraudulent == 1) & (A2_pred == 1));
A_Mdl_TF = sum((trainData.fraudulent == 0) & (A2_pred == 0));
% Calculate Precision for model 1
A_Mdl_Precision = (A_Mdl_TP / (A_Mdl_TP + A_Mdl_FP));
% Calculating Recall for model 1
A_Mdl_Recall = (A_Mdl_TP / (A_Mdl_TP + A_Mdl_FN));
% Calculating F1 for model 1
A_Mdl_f1 = 2/((1/A_Mdl_Recall)+(1/A_Mdl_Precision));
% Display values
disp("Model A - Tree Precision: " + A_Mdl_Precision);
disp("Model A - Tree Recall: " + A_Mdl_Recall);
disp("Model A - Tree f1: " + A_Mdl_f1);

% Create a confusion matrix to visualise class of outcome
confusionchart(trainData.fraudulent,A2_pred);

% Compute the posterior probabilities for DT
[~,A_score] = kfoldPredict(A2_mdl);
[A_X,A_Y,A_T,A_AUC] = perfcurve(trainData.fraudulent,A_score(:,2),1);
% Calculate ROC curve using the scores
disp("Model A - Tree AUC: " + A_AUC);

% Time to train the model
A2TrainHandle = @() crossval(A_mdl,'kfold', 10);
A2TimeTrain = timeit(A2TrainHandle);
disp("Model A - Tree train time: " + A2TimeTrain);

% Time to predict
A2TestHandle = @() kfoldPredict(A2_mdl);;
A2TimeTest = timeit(A2TestHandle);
disp("Model A - Tree pred time: " + A2TimeTest);

%% Model D - Fit a Naive Bayes Classification Model to our data:

% Partition, train and test will be reused from previous to keep the
% environement the same

% Create NB model setting distribution to Kernel not assuming normality
D_mdl = fitcnb(trainData,"fraudulent","DistributionNames","kernel");

% Calculating the resub loss of model on training set
D_errTrain =  resubLoss(D_mdl);
% Calculating model loss on the testing data
D_errTest = loss(D_mdl,testData);
% Print loss values
disp("Model D - NB Train Loss: " + D_errTrain);
disp("Model D - NB Test Loss: " + D_errTest);

% Predict fraud using our unseen test data
D_pred = predict(D_mdl,testData);

% Create 10 K-Fold partition on model D which used the training data
D2_mdl = crossval(D_mdl,'kfold', 10);
% Calculate the K-Fold loss
D2_errK = kfoldLoss(D2_mdl);
% Display the K-Fold loss
disp("Model D - NB K-Fold Loss: " + D2_errK);

% Predict the values on the K-Fold data
D2_pred = kfoldPredict(D2_mdl);

% Calculating FN, FP, TP & TF for model 2
D_Mdl_FN = sum((trainData.fraudulent == 1) & (D2_pred == 0));
D_Mdl_FP = sum((trainData.fraudulent == 0) & (D2_pred == 1));
D_Mdl_TP = sum((trainData.fraudulent == 1) & (D2_pred== 1));
D_Mdl_TF = sum((trainData.fraudulent == 0) & (D2_pred == 0));

% Calculate Precision for model 2
D_mdl_Precision = (D_Mdl_TP / (D_Mdl_TP + D_Mdl_FP));
% Calculating Recall for model 2
D_mdl_Recall = (D_Mdl_TP / (D_Mdl_TP + D_Mdl_FN));
% Calculating F1 for model 2
D_mdl_f1 = 2/((1/D_mdl_Recall)+(1/D_mdl_Precision));
% Display values
disp("Model D - NB Precision: " + D_mdl_Precision);
disp("Model D - NB Recall: " + D_mdl_Recall);
disp("Model D - NB f1: " + D_mdl_f1);

% Visulaise the outcome in confusion matrix
confusionchart(trainData.fraudulent,D2_pred);

% Compute the posterior probabilities for DT
[~,D_score] = kfoldPredict(D2_mdl);
% Calculate ROC curve using the scores
[D_X,D_Y,D_T,D_AUC] = perfcurve(trainData.fraudulent,D_score(:,2),1);
% Calculate ROC curve using the scores
disp("Model D - NB AUC: " + D_AUC);

% Time to train the model
D2TrainHandle = @() crossval(D_mdl,'kfold', 10);
D2TimeTrain = timeit(D2TrainHandle);
disp("Model D - Tree train time: " + D2TimeTrain);

% Time to predict
D2TestHandle = @() kfoldPredict(D2_mdl);
D2TimeTest = timeit(D2TestHandle);
disp("Model D - Tree pred time: " + D2TimeTest);

%% ROC Curve 1:
% Model A vs Model D: 
 
% Plot ROC Curve:
% Plot X Y for Tree
plot(A_X,A_Y)
% To plot another on top
hold on
%Plot X Y for NB
plot(D_X,D_Y)
% Add legend
legend('Decision Tree','Naive Bayes','Location','Best')
% Add X& Y labels
xlabel('False positive rate'); ylabel('True positive rate');
% Add Title
title('ROC Curves for Decision Tree VS Naive Bayes Classification');
% Off to not effect future plots
hold off
%% Trying to Improve the Models:

%% Predictor Importance for Tree model 1:

% Store predictor importance for model A
p = predictorImportance(A_mdl);
% Visualise the pridictors using a bar plot 
bar(p)
% Use variable names as labels for identification
xticklabels(data.Properties.VariableNames(1:end-1))

% Create a filter to keep relevant columns
toKeep = p > 0.00001;
% Insert filter in creation of a new dataset
predictorData = data(:,[toKeep]);
predictorData = [predictorData data(:,7:7)];

%% Principal Component Analysis on Data:
% Create new dataset removing the target column
features = [data{:,1:6} data{:,8:end}];
% Creating a variable with target column
target = categorical(data.fraudulent);
% Property saving column names
labels = data.Properties.VariableNames(1:end-1);

% Get the principal component coefficients
[pcs,scrs,~,~,pexp] = pca(features);
% Create a Pareto chart of the percent variance
pareto(pexp);
% Heat map of the absolute values of the first two columns
heatmap(abs(pcs(:,1:2)),"YDisplayLabels",labels);

% Create new data with PCA components and target column
new_columns = scrs(:,1:2);
t_data = table(new_columns);
pca_data = [t_data data(:,7:7)];
pca_data = splitvars(pca_data, 'new_columns');
pca_data.Properties.VariableNames = {'pca1' 'pca2' 'fraudulent'};

%% Removing Outliers from Data:
% Calculate the Mahalanobis distance between the pca components
mahal = mahal(pca_data.pca1,pca_data.pca2);
 
% Visualise in a scatter plot
scatter(pca_data.pca1,pca_data.pca2)
hold on
scatter(pca_data.pca1,pca_data.pca2,100,mahal,'o','filled')
hb = colorbar;
ylabel(hb,'Mahalanobis Distance')
hold off

% Creating a new dataset removing the outlier
% Create Mahal variable as a table
mahal_table = table(mahal);
% Joing with PCA data
pca_data = [pca_data mahal_table];
% Rename the columns
pca_data.Properties.VariableNames = {'pca1' 'pca2' 'fraudulent' 'Mahal'};
% Create a row filter to remove outliers
Keep = pca_data.Mahal < 2000;
% Add the filter as we create the new dataset
pca_data = pca_data([Keep],:);
% Drop the Mahal column as no longer needed
pca_data(:,4:4) = [];

%% Model F - Improving Model A using Resemble learning "Random Forest", PCA Componenets 1 & 2 and Optimising paramters:

% Partition the data using hold out method at 70-30 using pca data we created
cvpt_pca = cvpartition(pca_data.fraudulent,"HoldOut",0.3);
% Set training dataset
trainData_pca = pca_data(training(cvpt_pca),:);
% Set testing dataset
testData_pca = pca_data(test(cvpt_pca),:);

% Create a new Model using ensemble with method "Bag" for random forest
F_mdl = fitcensemble(trainData_pca,"fraudulent",'Method','Bag',"NumLearningCycles",30);

% Calculate the Resub loss for the training data
F_errTrain = resubLoss(F_mdl);
% Calculate the loss on the test data
F_errTest = loss(F_mdl,testData_pca);
% Print loss values
disp("Model F - RF Train Loss: " + F_errTrain )
disp("Model F - RF Test Loss: " + F_errTest)

% Predict the values on test data
F_pred = predict(F_mdl,testData_pca);

% Calculating FN, FP, TP & TF for model F
F_Mdl_FN = sum((testData_pca.fraudulent == 1) & (F_pred == 0));
F_Mdl_FP = sum((testData_pca.fraudulent == 0) & (F_pred == 1));
F_Mdl_TP = sum((testData_pca.fraudulent == 1) & (F_pred == 1));
F_Mdl_TF = sum((testData_pca.fraudulent == 0) & (F_pred == 0));
% Calculate Precision for model F
F_Mdl_Precision = (F_Mdl_TP / (F_Mdl_TP + F_Mdl_FP));
% Calculating Recall for model 1
F_Mdl_Recall = (F_Mdl_TP / (F_Mdl_TP + F_Mdl_FN));
% Calculating F1 for model 1
F_Mdl_f1 = 2/((1/F_Mdl_Recall)+(1/F_Mdl_Precision));
% Display values
disp("Model F - RF Precision: " + F_Mdl_Precision);
disp("Model F - RF Recall: " + F_Mdl_Recall);
disp("Model F - RF f1: " + F_Mdl_f1);

% Visualise using a Confusion Matrix
confusionchart(testData_pca.fraudulent,F_pred);

% Compute the posterior probabilities for DT
[~,F_score] = resubPredict(F_mdl);
% Calculate ROC curve using the scores
[F_X,F_Y,F_T,F_AUC] = perfcurve(trainData_pca.fraudulent,F_score(:,2),1);
% Calculate ROC curve using the scores
disp("Model F - RF K AUC: " + F_AUC);

%% Model G - Redo without PCA and keeping outliers and instead using Predictor Importance:
% No TP or FP detected by the model and an assumption that the PCA
% components are a result of this

% Partition the data using hold out method at 70-30 using Predictor Importance data we created
cvpt_PI = cvpartition(predictorData.fraudulent,"HoldOut",0.3);
% Set training dataset
trainData_PI = predictorData(training(cvpt_PI),:);
% Set testing dataset
testData_PI = predictorData(test(cvpt_PI),:);

% Create a new Model using ensemble with method "Bag" for random forest
G_mdl = fitcensemble(trainData_PI,"fraudulent",'Method','Bag',"NumLearningCycles",30);

% Calculate the Resub loss for the training data
G_errTrain = resubLoss(G_mdl);
% Calculate the loss on the test data
G_errTest = loss(G_mdl,testData_PI);
% Print loss values
disp("Model G - RF Train Loss: " + G_errTrain )
disp("Model G - RF Test Loss: " + G_errTest)

% Predict the values on test data
G_pred = predict(G_mdl,testData_PI);

% Create 10 K-Fold partition on model A which used the training data
G2_mdl = crossval(G_mdl,'kfold', 10);
% Calculate the K-Fold loss
G2_errK = kfoldLoss(G2_mdl);
% Display the K-Fold loss
disp("Model G - RF K-Fold Loss: " + G2_errK);

% Predict the values on the K-Fold data
G2_pred = kfoldPredict(G2_mdl);

% Calculating FN, FP, TP & TF for model G
G_Mdl_FN = sum((trainData_PI.fraudulent == 1) & (G2_pred == 0));
G_Mdl_FP = sum((trainData_PI.fraudulent == 0) & (G2_pred == 1));
G_Mdl_TP = sum((trainData_PI.fraudulent == 1) & (G2_pred == 1));
G_Mdl_TF = sum((trainData_PI.fraudulent == 0) & (G2_pred == 0));
% Calculate Precision for model G
G_Mdl_Precision = (G_Mdl_TP / (G_Mdl_TP + G_Mdl_FP));
% Calculating Recall for model G
G_Mdl_Recall = (G_Mdl_TP / (G_Mdl_TP + G_Mdl_FN));
% Calculating F1 for model G
G_Mdl_f1 = 2/((1/G_Mdl_Recall)+(1/G_Mdl_Precision));
% Display values
disp("Model G - RF Precision: " + G_Mdl_Precision);
disp("Model G - RF Recall: " + G_Mdl_Recall);
disp("Model G - RF f1: " + G_Mdl_f1);

% Visualise using a Confusion Matrix
confusionchart(trainData_PI.fraudulent,G2_pred);

% Compute the posterior probabilities for DT
[~,G_score] = resubPredict(G_mdl);
% Calculate ROC curve using the scores
[G_X,G_Y,G_T,G_AUC] = perfcurve(trainData_PI.fraudulent,G_score(:,2),1);
% Calculate ROC curve using the scores
disp("Model G - RF K AUC: " + G_AUC);

% Time to train the model
G2TrainHandle = @() crossval(G_mdl,'kfold', 10);
G2TimeTrain = timeit(G2TrainHandle);
disp("Model G - RF train time: " + G2TimeTrain);

% Time to predict
G2TestHandle = @() kfoldPredict(G2_mdl);;
G2TimeTest = timeit(G2TestHandle);
disp("Model G - RF pred time: " + G2TimeTest);

%% Model I - Improving Model D by Optimising Hypper paramters:

% We use the same partition, test and train sets as model 1 & 2: 

% Create new improved model
I_mdl = fitcnb(trainData,"fraudulent","DistributionNames","kernel","Width",3.7688e+05);
% Calculating the resub loss of model on training set
I_errTrain = resubLoss(I_mdl);
% Calculating model loss on the testing data
I_errTest = loss(I_mdl,testData);
% Print loss values

disp("Model I - NB Training Loss: " + I_errTrain);
disp("Model I - NB Testing Loss: " + I_errTest);

% Predicting model outputs
I_pred = predict(I_mdl,testData);

% Create 10 K-Fold partition on model A which used the training data
I2_mdl = crossval(I_mdl,'kfold', 10);
% Calculate the K-Fold loss
I2_errK = kfoldLoss(I2_mdl);
% Display the K-Fold loss
disp("Model I - NB K-Fold Loss: " + I2_errK);

% Predict the values on the K-Fold data
I2_pred = kfoldPredict(I2_mdl);

% Calculating FN, FP, TP & TF for model I
I_mdl_FN = sum((trainData.fraudulent == 1) & (I2_pred == 0));
I_mdl_FP = sum((trainData.fraudulent == 0) & (I2_pred == 1));
I_mdl_TP = sum((trainData.fraudulent == 1) & (I2_pred == 1));
I_mdl_TF = sum((trainData.fraudulent == 0) & (I2_pred == 0));
% Calculate Precision for model I
I_mdl_Precision = (I_mdl_TP / (I_mdl_TP + I_mdl_FP));
% Calculating Recall for model I
I_mdl_Recall = (I_mdl_TP / (I_mdl_TP + I_mdl_FN));
% Calculating F1 for model I
I_mdl_f1 = 2/((1/I_mdl_Recall)+(1/I_mdl_Precision));
% Display values
disp("Model I - NB Precision: " + I_mdl_Precision);
disp("Model I - NB Recall: " + I_mdl_Recall);
disp("Model I - NB f1: " + I_mdl_f1);

% Visulaise the outcome in confusion matrix
confusionchart(trainData.fraudulent,I2_pred);

% Compute the posterior probabilities for NB
[~,I_score] = kfoldPredict(I2_mdl);
% Calculate ROC curve using the scores
[I_X,I_Y,I_T,I_AUC] = perfcurve(trainData.fraudulent,I_score(:,2),1);
% Calculate ROC curve using the scores
disp("Model I - NB K AUC: " + I_AUC);

% Time to train the model
I2TrainHandle = @() crossval(I_mdl,'kfold', 10);
I2TimeTrain = timeit(I2TrainHandle);
disp("Model I - NB train time: " + I2TimeTrain);

% Time to predict
I2TestHandle = @() kfoldPredict(I2_mdl);;
I2TimeTest = timeit(I2TestHandle);
disp("Model I - NB pred time: " + I2TimeTest);

%% Model G vs Model I: 
%% ROC Curve 2:
% Model G vs Model I:

% Plot ROC Curve:
% Plot X Y for Tree
plot(G_X,G_Y)
% To plot another on top
hold on
%Plot X Y for NB
plot(I_X,I_Y)
% Add legend
legend('Decision Tree','Naive Bayes','Location','Best')
% Add X& Y labels
xlabel('False positive rate'); ylabel('True positive rate');
% Add Title
title('ROC Curves for Decision Tree VS Naive Bayes Classification');
% Off to not effect future plots
hold off

%% Prune  - Model P to see effect of Pruning:
% Prune the tree model to reduce over fitting
P_mdl = prune(A_mdl,"Level",1);
 
% Calculate the loss erros again
P_errTrain = resubLoss(P_mdl);
P_errTest = loss(P_mdl,testData);
% Display loss values again
disp("Model P - Tree Pruned Training Loss: " + P_errTrain);
disp("Model P - Tree Pruned Test Loss: " + P_errTest);

% text description
view(P_mdl);
% Graphic description
view(P_mdl,'mode','graph');

% Predict the values on the test data
P_pred = predict(P_mdl,testData);

% Create 10 K-Fold partition on model P which used the training data
P2_mdl = crossval(P_mdl,'kfold', 10);
% Calculate the K-Fold loss
P2_errK = kfoldLoss(P2_mdl);
% Display the K-Fold loss
disp("Model P - Tree Prune K-Fold Loss: " + P2_errK);

% Predict the values on the K-Fold data
P2_pred = kfoldPredict(P2_mdl);

% Calculating FN, FP, TP & TF for model P
P_Mdl_FN = sum((trainData.fraudulent == 1) & (P2_pred == 0));
P_Mdl_FP = sum((trainData.fraudulent == 0) & (P2_pred == 1));
P_Mdl_TP = sum((trainData.fraudulent == 1) & (P2_pred == 1));
P_Mdl_TF = sum((trainData.fraudulent == 0) & (P2_pred == 0));
% Calculate Precision for model P
P_Mdl_Precision = (P_Mdl_TP / (P_Mdl_TP + P_Mdl_FP));
% Calculating Recall for model 1
P_Mdl_Recall = (P_Mdl_TP / (P_Mdl_TP + P_Mdl_FN));
% Calculating F1 for model P
P_Mdl_f1 = 2/((1/P_Mdl_Recall)+(1/P_Mdl_Precision));
% Display values
disp("Model P - Pruned Tree Precision: " + P_Mdl_Precision);
disp("Model P - Pruned Tree Recall: " + P_Mdl_Recall);
disp("Model P - Pruned Tree f1: " + P_Mdl_f1);

% Create a confusion matrix to visualise class of outcome
confusionchart(trainData.fraudulent,P2_pred);
