function FN_TreeBaggerAnalysis( DATA,DATA_TAGS,NUMBER_OF_TREES )

[G GN] = grp2idx(DATA_TAGS);  % Reduce character tags to numeric grouping

Training_model = TreeBagger(NUMBER_OF_TREES,...
    DATA,...
    G,...
    'Method', 'classification',...
    'OOBVarImp','on',...
    'OOBPred','on',...
    'NPrint',1);


figure
plot(oobError(Training_model));
xlabel 'Number of Grown Trees';
ylabel 'Out-of-Bag Mean Squared Error';


figure
bar(Training_model.OOBPermutedVarDeltaError);
xlabel 'Feature Number' ;
ylabel 'Out-of-Bag Feature Importance';
idxvar = find(Training_model.OOBPermutedVarDeltaError>0.7)
idxCategorical = find(iscategorical(idxvar)==1);


b5v = TreeBagger(NUMBER_OF_TREES,DATA(:,idxvar),G,'Method','R',...
    'OOBVarImp','On','CategoricalPredictors',idxCategorical);
figure
plot(oobError(b5v));
xlabel 'Number of Grown Trees';
ylabel 'Out-of-Bag Mean Squared Error';
figure
 bar(b5v.OOBPermutedVarDeltaError);
 xlabel 'Feature Index';
 ylabel 'Out-of-Bag Feature Importance';
% 
% b5v = fillProximities(b5v);
% figure
% hist(b5v.OutlierMeasure);
% xlabel 'Outlier Measure';
% ylabel 'Number of Observations';
% figure(8);
% [~,e] = mdsProx(b5v,'Colors','K');
% xlabel 'First Scaled Coordinate';
% ylabel 'Second Scaled Coordinate';
end

