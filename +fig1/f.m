function f
%% This function plots a panel from the manuscript (Figure 1f)
figure;
set(gcf, 'position', get(gcf, 'position').*[1 1 2, 1])

subplot(1,2,1);
blks = spatialAnalysis('PC022', 'behavior', 0, 1, '');
blks.blks = prc.filtBlock(blks.blks, blks.blks.tri.stim.visContrast ~= 0.06);
blks.viewGLMFits('simpLogSplitVSplitA', [],'log', 1)
ylabel('p(Rightward)/p(Leftward)')
set(gca, 'YTickLabel', [0.01, 0.1, 1, 10, 100])
xlabel('Visual contrast')
axis square;
title('Example mouse 1')

subplot(1,2,2);
blks = spatialAnalysis('PC051', 'behavior', 0, 1, '');
blks.blks = prc.filtBlock(blks.blks, blks.blks.tri.stim.visContrast ~= 0.06);
blks.viewGLMFits('simpLogSplitVSplitA', [],'log', 1)
xlabel('Visual contrast')
set(gca, 'ycolor', 'none');
axis square;
title('Example mouse 2')
end