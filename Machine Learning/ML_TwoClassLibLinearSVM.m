function [ Confusion,...
    Accuracy,...
    Probablity_estimates,...
    Training_model,...
    Final_decision ,...
    Actual_answer  ]...
    = ML_TwoClassLibSVM( DATA,TESTIDX,TRAINIDX,G,~,VERBOSE )




Training_model = cell(1,1);
predictions = zeros(sum(TESTIDX),numel(Training_model)); %# store binary predictions

Training_model{1} = train( G(TRAINIDX), sparse(DATA(TRAINIDX,:)));
[predictions(:,1), accuracy, Probablity_estimates] =  predict(G(TESTIDX), sparse(DATA(TESTIDX,:)), Training_model{1});
Accuracy = accuracy(1);
GG = G;
uG = unique(G);



GG(G == uG(2)) = -1;
GG(G == uG(1)) = 1; 
[~,~,~,AUC] = perfcurve(GG(TESTIDX),Probablity_estimates,1);

Final_decision = mode(predictions,2);   %# votipredng: clasify as the class receiving most votes
Final_decision = Final_decision - 1;
Actual_answer = G - 1;    %Gotta minus one to fit the confusion range

Confusion = confusionmat(Final_decision, Actual_answer(TESTIDX));
[~,~,~,~,Accuracy,~] = ...
    MISC_PlotConfusion(Confusion,[1,2],0);

if VERBOSE
    disp(['AUC : ',num2str(AUC)]);
    disp(Confusion);
end
disp(['Accuracy : ',num2str(Accuracy * 100)]);

end




