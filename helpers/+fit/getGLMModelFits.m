function getGLMModelFits
%% Saves the logliklihood for 5 fold CV model fits so they can be accessed fast
% Model fits for the standard behaviour data
models2fit = {'biasOnly';'visOnly';'audOnly';'simpLog';'simpLogSplitV';'simpLogSplitA';'simpLogSplitVSplitA';...
    'fullEmp';'simpEmp';'visOnlyEmp';'audOnlyEmp'; 'simpLogSplitVSplitAUnisensory';...
    'simpLogSplitVSplitAAudDom'; 'simpLogSplitVSplitAAudExtraDom'; 'simpLogSplitVSplitASplitT'};  

crossVal = 5;

allBlks = spatialAnalysis('all', 'behavior', 1, 0, '');
sStart = spatialAnalysis('all', 'behavior', 0, 1, '');

% Remove instances of 6% contrast which were used in a small set of mice
allBlks.blks = prc.filtBlock(allBlks.blks, allBlks.blks.tri.stim.visContrast ~= 0.06);
for i = 1:length(sStart.blks)
    sStart.blks(i) = prc.filtBlock(sStart.blks(i), sStart.blks(i).tri.stim.visContrast ~= 0.06);
end
allBlks.blks.exp.subject{1} = 'Combined';
sStart.blks(end+1) = allBlks.blks;

saveDir = [prc.pathFinder('processeddirectory') '\XSupData\GLMFits2Behavior'];
if ~exist(saveDir, 'dir'); mkdir(saveDir); end

for i = 1:length(models2fit)
    fprintf('Currently fitting model %s \n', models2fit{i});
    s = sStart;
    s.viewGLMFits(models2fit{i}, crossVal);
    save([saveDir '\' models2fit{i} '_Cross' num2str(crossVal) '.mat'], 's');
    close;
end

s = sStart;
s.viewGLMFits('biasOnly');
save([saveDir '\BiasOnlyPerformance.mat']);
close;

s = sStart;
s.viewGLMFits('fullEmp');
save([saveDir '\FullEmpMaxPerformance.mat']);
close;

%%
% Model fits for the 5-auditory condition behaviour data
models2fit = {'simpLogSplitVEmpA';'fullEmp'};  
sStart = spatialAnalysis('PC013', 'aud5', 1, 0, '');

for i = 1:length(models2fit)
    fprintf('Currently fitting model %s \n', models2fit{i});
    s = sStart;
    s.viewGLMFits(models2fit{i}, crossVal);
    save([saveDir '\' models2fit{i} '_5Aud_Cross' num2str(crossVal) '.mat'], 's');
    close;
end

s = sStart;
s.viewGLMFits('biasOnly');
save([saveDir '\BiasOnlyPerformance_5Aud.mat']);
close;

s = sStart;
s.viewGLMFits('fullEmp');
save([saveDir '\FullEmpMaxPerformance_5Aud.mat']);
close;
end
