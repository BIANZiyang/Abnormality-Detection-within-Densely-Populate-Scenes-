SetupVariables;

DATA_VIDEO_CHOSENSET = DATA_VIDEO_KTH;

VideoList = FN_PopulateStandardList(DATA_VIDEO_CHOSENSET.dir,10);

Param_GLCM = Param_GLCM_Default;

Param_EdgeCardinality = Param_EdgeCardinality_Default;

Param_PixelDifference=  Param_PixelDifference_Default;

WindowSize = 24;
WindowSkip = 12;
ImageResize = 0.5;

% Extract Descriptors
[GEPNonPCADescriptors,GEPDescriptors,GEPTags,GEPFlowList,GEPGroup]...
    = FN_GEPDescriptor(VideoList,...
    DATA_VIDEO_CHOSENSET,...
    Param_GLCM,...
    Param_EdgeCardinality,...
    Param_PixelDifference,...
    WindowSize,...
    WindowSkip,...
    ImageResize);

% Perform Classification
[RAND_FOREST,LINEAR_SVM] = ...
    FN_CrossValidationTesting( GEPDescriptors,GEPGroup,GEPTags,true,true );
