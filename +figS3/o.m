function o
%% This function plots a panel from the manuscript (Figure S3o)
figure;
blks = spatialAnalysis('PC050', 'behavior', 0, 1, '');
blks.blks = prc.filtBlock(blks.blks, blks.blks.tri.stim.visContrast ~= 0.06);
blks.viewGLMFits('simpLogSplitVSplitA', [],'log', 1)
ylabel('p(Rightward)/p(Leftward)')
set(gca, 'YTickLabel', [0.01, 0.1, 1, 10, 100])
xlabel('Visual contrast')
axis square;
title('')

end