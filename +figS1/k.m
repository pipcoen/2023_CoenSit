function k
%% This function plots a panel from the manuscript (Figure S1k)
glmBlk = spatialAnalysis('PC013', 'aud5', 1, 1);

figure;
glmBlk.viewGLMFits('simpLogSplitVEmpA', [],'log', 1);
ylabel('p(Rightward)/p(Leftward)')
xlabel('Visual contrast')
ylim([-2.9 2.9])
set(gca, 'YTickLabel', [0.01, 0.1, 1, 10, 100])
axis square;
end