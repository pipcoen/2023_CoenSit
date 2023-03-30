function f
%% This function plots a panel from the manuscript (Figure S2f)
figure;
set(gcf, 'position', get(gcf, 'position').*[1 1 2, 1])

subplot(1,2,1);
% "normalize" block--removes timeouts, laser, and nan-response trials
glmBlk = spatialAnalysis('all', 'behavior', 1, 0,'');
glmBlk.blks = spatialAnalysis.getBlockType(glmBlk.blks, 'norm');
% Remove instances of 6% contrast which were used in a small set of mice
glmBlk.blks = prc.filtBlock(glmBlk.blks, glmBlk.blks.tri.stim.visContrast ~= 0.06);

glmBlk.viewGLMFits('simpLogSplitVSplitA', [],'controllog', 3);
pow2use = glmBlk.glmFit{1}.prmFits(4);

%%
glmBlk.viewGLMFits('simpEmp', [],'log', 10, pow2use)
ylabel('p(Rightward)/p(Leftward)')
xlabel('Visual contrast')
set(gca, 'YTickLabel', [0.01, 0.1, 1, 10, 100])
axis square;
title('Nonparametric additive')


subplot(1,2,2);
loglikDat = fit.compareSpecifiedModels({'simpLogSplitVSplitA_Cross5'; 'simpEmp_Cross5'});
hold on
xDat = loglikDat.simpLogSplitVSplitA_Cross5;
yDat = loglikDat.simpEmp_Cross5;
xlim([0 0.55]);
ylim([0 0.55]);
xlabel('Additive model (bits/trial)')
ylabel('Unconstrained additive model (bits/trial)')
axis square;
plot([0,0.55], [0,0.55], '--k')

scatter(xDat, yDat, 50, 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.5, 0.5, 0.5]);

[~, pVal] = ttest(xDat(1:end-1), yDat(1:end-1));
text(0.4, 0.1, ['pVal ~ ' num2str(pVal)]);
end