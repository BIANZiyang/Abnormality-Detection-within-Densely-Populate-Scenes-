function [RAND_FOREST,LINEAR_SVM] = FN_CrossValidationTesting( DATA,DATA_GROUP,DATA_TAGS )
global RANDOM_FOREST_VERBOSE;
global RANDOM_FOREST_TREES;
global RANDOM_FOREST_VERBOSE_MODEL;
global LINEAR_SVM_VERBOSE;

if isempty(LINEAR_SVM_VERBOSE)
    LINEAR_SVM_VERBOSE = true;
end

if isempty(RANDOM_FOREST_VERBOSE_MODEL)
    RANDOM_FOREST_VERBOSE_MODEL = false;
end

if isempty(RANDOM_FOREST_VERBOSE)
    RANDOM_FOREST_VERBOSE = false;
end

if isempty(RANDOM_FOREST_TREES)
    RANDOM_FOREST_TREES = 50;
end

Folds = max(cell2mat(DATA_GROUP));
DataSplit = cell(Folds,5);

RandomForest_Confusion            = cell(Folds,1);
RandomForest_Accuracy             = zeros(Folds,1);
RandomForest_Probablity_estimates = cell(Folds,1);
RandomForest_Training_model       = cell(Folds,1);
RandomForest_Final_decision       = cell(Folds,1);
RandomForest_Actual_answer        = cell(Folds,1);

LinearSVM_Confusion            = cell(Folds,1);
LinearSVM_Accuracy             = zeros(Folds,1);
LinearSVM_Probablity_estimates = cell(Folds,1);
LinearSVM_Training_model       = cell(Folds,1);
LinearSVM_Final_decision       = cell(Folds,1);
LinearSVM_Actual_answer        = cell(Folds,1);


[G GN] = grp2idx(DATA_TAGS);  % Reduce character tags to numeric grouping

for k = 1: Folds
    disp(['Starting Test ',num2str(k)]);
    testData = find([DATA_GROUP{:}]'== k);
    TESTIDX = false(length(DATA_GROUP),1);
    TESTIDX(testData) = true;
    TRAINIDX = ~TESTIDX;
    
    % Save group assignments into a
    DataSplit{k,1} = k;
    DataSplit{k,2} = TRAINIDX;
    DataSplit{k,3} = TESTIDX;
    DataSplit{k,4} = G;
    DataSplit{k,5} = GN;
    
    %% TEST USING LINEAR SVM
    [ LinearSVM_Confusion{k},...
        LinearSVM_Accuracy(k),...
        LinearSVM_Probablity_estimates{k},...
        LinearSVM_Training_model{k},...
        LinearSVM_Final_decision{k},...
        LinearSVM_Actual_answer{k} ]...
        = ML_TwoClassLibLinearSVM(DATA ,TESTIDX,TRAINIDX,G,GN,LINEAR_SVM_VERBOSE );
    
    
    
    %% TEST RANDOM FOREST
    [RandomForest_Confusion{k},...
        RandomForest_Accuracy(k),...
        RandomForest_Probablity_estimates{k},...
        RandomForest_Training_model{k},...
        RandomForest_Final_decision{k},...
        RandomForest_Actual_answer{k} ] = ML_TwoClassForest(DATA,...
        TESTIDX,...
        TRAINIDX,...
        G,...
        GN,...
        RANDOM_FOREST_TREES,...
        RANDOM_FOREST_VERBOSE,...
        RANDOM_FOREST_VERBOSE_MODEL);
end

RAND_FOREST = {RandomForest_Confusion,...
    RandomForest_Accuracy,...
    RandomForest_Probablity_estimates,...
    RandomForest_Training_model,...
    RandomForest_Final_decision,...
    RandomForest_Actual_answer};

LINEAR_SVM = {LinearSVM_Accuracy_Confusion,...
    LinearSVM_Accuracy,...
    LinearSVM_Accuracy_Probablity_estimates,...
    LinearSVM_Accuracy_Training_model,...
    LinearSVM_Accuracy_Final_decision,...
    LinearSVM_Accuracy_Actual_answer};


end

