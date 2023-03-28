function g
%% This function plots a panel from the manuscript (Figure 1g)
% Identify stimulus conditions that were not present for all mice
blks = spatialAnalysis('all', 'behavior', 0, 1,'');
allParametersAV = cell2mat(arrayfun(@(x) x.exp.conditionParametersAV{1}, blks.blks, 'uni', 0));
[uniParametersAV, ~, rowIdx] = unique(allParametersAV, 'rows');
condFreq = histcounts(rowIdx,1:max(rowIdx)+1)';
cond2Use = uniParametersAV(condFreq==max(condFreq),:);
%%
% "normalize" block--removes timeouts, laser, and nan-response trials
glmBlk = spatialAnalysis('all', 'behavior', 1, 0,'');
glmBlk.blks = spatialAnalysis.getBlockType(glmBlk.blks, 'norm');
audDiff = glmBlk.blks.tri.stim.audDiff;
visDiff = glmBlk.blks.tri.stim.visDiff;
% Remove stimulus conditions that were not present for all mice
glmBlk.blks = prc.filtBlock(glmBlk.blks, ismember([audDiff visDiff], cond2Use, 'rows'));
numExp = glmBlk.blks.tot.experiments;
glmBlk.blks.exp.conditionParametersAV = repmat(glmBlk.blks.exp.conditionParametersAV(end),numExp,1);

% Plot fit
figure;
% Fit with 10 random subsamples of combined mice to equalize across mice
glmBlk.viewGLMFits('simpLogSplitVSplitA', [],'log', 10)
ylabel('p(Rightward)/p(Leftward)')
xlabel('Visual contrast')
set(gca, 'YTickLabel', [0.01, 0.1, 1, 10, 100])
axis square;
title('17 mice')
end