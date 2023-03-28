function f
%% This function plots a panel from the manuscript (Figure S1f)
blks = spatialAnalysis('all', 'behavior', 1, 0, '');
% Remove instances of 6% contrast which were used in a small set of mice
blks = prc.filtBlock(blks.blks, blks.blks.tri.stim.visContrast ~= 0.06);
%"normalize" block--removes timeouts, laser, and nan-response trials
blks = spatialAnalysis.getBlockType(blks, 'norm');

%extract reaction times
allRTs = blks.tri.outcome.reactionTime;

%%
figure;
set(gcf, 'position', get(gcf, 'Position').*[1,1,0.6,1])
histogram(allRTs, 0:0.01:1.5, "EdgeColor","none","FaceColor",'k',"FaceAlpha",1);
xline(0.5, '--k');
box off;
xlim([0 1.5]);
ylabel('Number of trials')
xlabel('Reaction time (s)')
end