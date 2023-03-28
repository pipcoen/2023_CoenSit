function b
%% This function plots a panel from the manuscript (Figure S4b)
loadDir = [prc.pathFinder('processeddirectory') 'XSupData\'];
load([loadDir 'figS4bInactResultsForChoice_IndiMice'], 'inactResultsForChoice')
nMice = size(inactResultsForChoice.laserOnData,2);
%%
figure;
set(gcf, 'position', get(gcf, 'position').*[1 0.2 2, 2])
scanPlot.gridXY = inactResultsForChoice.gridXY{1};
scanPlot.sigLevels = [10^-3; 10^-4; 10^-5];
yLabs = {'Visual trials'; 'Auditory trials'; 'Coherent trials'; 'Conflict trials'};
for i = 1:length(inactResultsForChoice.subsets)
    for j = 1:nMice

        nShuffles = size(inactResultsForChoice.shuffLaserOffData{i,j},3);
        contData = inactResultsForChoice.meanContEffects{i,j};
        shuffleData = double(inactResultsForChoice.shuffLaserOnData{i,j} - inactResultsForChoice.shuffLaserOffData{i,j});
        sortedData = cellfun(@squeeze, num2cell(sort(abs(cat(3,shuffleData, contData)),3,'descend'),3), 'uni', 0);
        scanPlot.pVals = cell2mat(arrayfun(@(x,y) max([find(x==y{1},1) nan])./nShuffles, abs(contData), sortedData,'uni', 0));
        scanPlot.data = contData;

        %%Plot the data in the "scanPlot" structure.
        subplot(4, nMice,j+(i-1)*5)
       
        if i == 1 && j == 1; scanPlot.legendOff = 0; else; scanPlot.legendOff = 1; end

        plt.scanningBrainEffects(scanPlot);

        if i == 1 && j == 1
            scanPlot.colorBar = colorbar;
            scanPlot.colorBar.Position = scanPlot.colorBar.Position.*[0.5,1,1,1];
            set(scanPlot.colorBar, 'Ticks', [-0.6, 0 0.6])
            scanPlot.colorBar.Label.String = '\Delta Rightward';
            scanPlot.colorBarLimits = [-0.6 0.6];
            scanPlot = rmfield(scanPlot, 'colorBar');
        end
        if i == 1
            text(0, 4.5, sprintf('Mouse %d', j), 'FontWeight','bold', 'HorizontalAlignment', 'center');
        end
        if j == 1
            text(-6, -5, yLabs{i}, 'FontWeight','bold', 'VerticalAlignment', 'middle', 'Rotation', 90);
        end
        ylim([-5.5 5]);
    end
end
end