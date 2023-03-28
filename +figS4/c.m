function c
%% This function plots a panel from the manuscript (Figure S4c)
loadDir = [prc.pathFinder('processeddirectory') 'XSupData\'];
load([loadDir 'figS4bInactResultsForChoice_IndiMice'], 'inactResultsForChoice')
%%
subsets = inactResultsForChoice.subsets;
nMice = size(inactResultsForChoice.laserOnData,2);
[ipsiOut, contraOut] = deal(zeros(nMice, length(subsets)));
for j = 1:nMice
    for i = 1:length(subsets)        
        contData = inactResultsForChoice.meanContEffects{i,j};
        contData = contData(inactResultsForChoice.gridXY{1}{2}==5.5);
        ipsiOut(j,i) = contData(2);
        contraOut(j,i) = contData(end-1);
    end
end
%%
figure; hold on;
xDat = 1:4;
for j = 1:2
    if j == 1; plotData = ipsiOut; pltCol = [0.9290, 0.6940, 0.1250]; end
    if j == 2; plotData = contraOut; pltCol = [189, 48, 255]/255; end
    for i = 1:3
        plot([xDat(i),xDat(i+1)]+[0.1, -0.1], plotData(:,i:i+1), ...
            'Color', pltCol, 'linewidth', 1);

        meanLine = mean(plotData);
        plot([xDat(i),xDat(i+1)]+[0.1, -0.1], meanLine(i:i+1), 'Color', pltCol, 'linewidth', 2);
    end
end
box off;
xlabel('Stimulus Type')
ylabel('\Delta Fraction ipsi choices')
ylim([-0.3 0.3]);
xlim([0.8 4.2])
set(gca, 'xTick', xDat, 'xTickLabel', {'A', 'V', 'A=V', 'A≠V'}, 'YTick', [-0.3 0 0.3])
yline(0, '--k')

text(2, 0.25,'Ipsi to stimulus', 'color', [0.9290, 0.6940, 0.1250], 'FontSize',12, 'FontWeight','bold');
text(2, 0.22,'Contra to stimulus', 'color', [189, 48, 255]/255, 'FontSize',12, 'FontWeight','bold');
