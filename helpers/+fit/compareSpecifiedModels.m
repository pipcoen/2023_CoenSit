function loglikDat = compareSpecifiedModels(fileNames)
%% This function plots a panel from the manuscript (Figure 1h)
loadDir = [prc.pathFinder('processeddirectory') 'XSupData\GLMFits2Behavior'];
load([loadDir '\BiasOnlyPerformance.mat'], 's');
loglikDat.subjects = arrayfun(@(x) cell2mat(unique(x.exp.subject)'), s.blks, 'uni', 0);
minPerformance = cell2mat(cellfun(@(x) x.logLik, s.glmFit, 'uni', 0))*-1;

for i = 1:length(fileNames)
    load([loadDir '\' fileNames{i}], 's');
    logLik = cell2mat(cellfun(@(x) mean(x.logLik), s.glmFit, 'uni', 0));
    loglikDat.(fileNames{i}) = logLik*-1 - minPerformance;
end
end
