function l
%% This function plots a panel from the manuscript (Figure S1l)
fileNames = {'simpLogSplitVEmpA_5Aud_Cross5'; 'fullEmp_5Aud_Cross5'};
loadDir = [prc.pathFinder('processeddirectory') 'XSupData\GLMFits2Behavior'];
load([loadDir '\BiasOnlyPerformance_5Aud.mat'], 's');

loglikDat.subjects = arrayfun(@(x) cell2mat(unique(x.exp.subject)'), s.blks, 'uni', 0);
minPerformance = cell2mat(cellfun(@(x) x.logLik, s.glmFit, 'uni', 0))*-1;
%%
for i = 1:length(fileNames)
    load([loadDir '\' fileNames{i}], 's');
    logLik = cell2mat(cellfun(@(x) (x.logLik), s.glmFit, 'uni', 0));
    loglikDat.(fileNames{i}) = logLik*-1 - minPerformance;
end

%%
figure;
hold on;
set(gcf, 'position', get(gcf, 'Position').*[1,1,0.5,1])
set(gca, 'yTick', [0 0.5], 'xTick', [0.25 0.75], 'XTickLabel', {'Add'; 'Full'})
ylim([0,0.5])
xlim([0 1])

plot(0.75 + rand(5,1)*0.05, loglikDat.fullEmp_5Aud_Cross5, '.k', 'MarkerSize', 20)
plot(0.25 + rand(5,1)*0.05, loglikDat.simpLogSplitVEmpA_5Aud_Cross5, '.k', 'MarkerSize', 20)
ylabel('Loglikelihood (bits/trial)')
xlabel('Model Type')

[~, pVal] = ttest2(loglikDat.fullEmp_5Aud_Cross5, loglikDat.simpLogSplitVEmpA_5Aud_Cross5);
pVal = round(pVal, 2, 'significant');
text(0.75, 0.05, ['P>' num2str(pVal)]);
end