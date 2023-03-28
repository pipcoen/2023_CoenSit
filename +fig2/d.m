function d
%% This function plots a panel from the manuscript (Figure 2d)
loadDir = [prc.pathFinder('processeddirectory') 'XSupData\'];
load([loadDir 'fig2b-eInactResultsForChoice'], 'inactResultsForChoice')
%%
idx = find(strcmp(inactResultsForChoice.subsets, 'CohL'));
nShuffles = size(inactResultsForChoice.shuffLaserOffData{idx},3);
contData = inactResultsForChoice.meanContEffects{idx};
shuffleData = double(inactResultsForChoice.shuffLaserOnData{idx} - inactResultsForChoice.shuffLaserOffData{idx});
sortedData = cellfun(@squeeze, num2cell(sort(abs(cat(3,shuffleData, contData)),3,'descend'),3), 'uni', 0);


%% Plot the data in the "scanPlot" structure.
scanPlot.pVals = cell2mat(arrayfun(@(x,y) max([find(x==y{1},1) nan])./nShuffles, abs(contData), sortedData,'uni', 0));
scanPlot.data = contData;
scanPlot.addTrialNumber = 0;
sigLevels = (10.^(-2:-1:-10))';
lastSigLevel = find(sigLevels>min(scanPlot.pVals(:)),1,'last');
scanPlot.sigLevels = sigLevels(max([1 lastSigLevel-2]):lastSigLevel);
scanPlot.title = inactResultsForChoice.subsets{idx};
scanPlot.gridXY = inactResultsForChoice.gridXY{1};
scanPlot.colorBarLimits = [-0.6 0.6];

%%
figure;
set(gcf, 'position', get(gcf, 'position').*[1 1 0.5, 0.5])
scanPlot.colorBar = colorbar;
plt.scanningBrainEffects(scanPlot);
scanPlot.colorBar.Position = scanPlot.colorBar.Position.*[1.1,1,0.5,0.6];
set(scanPlot.colorBar, 'Ticks', [-0.6, 0 0.6])
text(0,-5.5, 'Coherent trials', 'HorizontalAlignment', 'center')
title('');
end