global RANDOM_FOREST_VERBOSE;
    RANDOM_FOREST_VERBOSE = true;
global RANDOM_FOREST_TREES;
    RANDOM_FOREST_TREES = 10;
global RANDOM_FOREST_VERBOSE_MODEL;
    RANDOM_FOREST_VERBOSE_MODEL = false;
global LINEAR_SVM_VERBOSE;
    LINEAR_SVM_VERBOSE = true;
    
    
DATA_GLCM = 'F:\DATA\DATA-GLCMMEX';
%DATA_GLCM = 'C:\Users\kaelon\Documents\DATA\DATA-GLCM';
DATA_VIDEO_UFC = struct('dir','F:\Videos\UFC\Normal_Abnormal_Crowd Converted',...
    'name','UFC');

DATA_VIDEO_ALLCROWDS = struct('dir','F:\Videos\ALLCROWDS\Crowd-Activity'...
    ,'name','ALLCROWDS');

DATA_VIDEO_CVD = struct('dir','F:\Videos\CVD'...
    ,'name','CVD');

DATA_VIDEO_KTH = struct('dir','Z:\KTH'...
    ,'name','KTH');

DATA_VIDEO_UMNSCENE1 = struct('dir','F:\Videos\UMN Scene 1',...
    'name','UMN_SCENE1');
DATA_VIDEO_UMNSCENE2 = struct('dir','F:\Videos\UMN Scene 2',...
    'name','UMN_SCENE2');
DATA_VIDEO_UMNSCENE3 = struct('dir','F:\Videos\UMN Scene 3',...
    'name','UMN_SCENE3');

% Default GLCM parameters
Param_GLCM_Default = struct('baseoffsets',[1 0; -1 0; 0 1;0 -1],...
    'graylevel',64,...
    'pyramid',[1 1],...
    'range',[1 2 3 4]);

Param_EdgeCardinality_Default = struct('type','log');

Param_PixelDifference_Default = struct('framejump',[1]);