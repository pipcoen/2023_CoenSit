function d
%% This function plots a panel from the manuscript (Figure S4d)
loadDir = [prc.pathFinder('processeddirectory') 'XSupData\'];
load([loadDir 'figS4dInactResultsForModelGamma'], 'inactResultsForModel')
inRes = inactResultsForModel;
%%
figure
scanPlot.gridXY = inRes.gridXY;
scanPlot.colorBarLimits = [-10 10];
scanPlot.sigLevels = [0.01; 0.001; 0.0001];
prmOrder = {'N', 'visScaleIpsi', 'visScaleConta', 'Bias', 'audScaleIpsi', 'audScaleContra'};
prmTitle = {'\gamma', 'Vi', 'Vc', 'b', 'Ai', 'Ac'};
for i = 1:length(prmOrder)
    subplot(2,3,i)
    idx = find(strcmp(prmOrder{i}, inRes.prmLabels));
    nShuffles = size(inRes.deltaParams{1},1) - inRes.normEstRepeats;
    inRes.contParams(cellfun(@isempty, inRes.contParams)) = deal({nan*ones(max(max(cellfun(@length, inRes.contParams))),6)});
    contData = cellfun(@(x) mean(x(1:inRes.normEstRepeats,idx), 'omitnan'),inRes.deltaParams);
    stdData = cellfun(@(x) std(x(inRes.normEstRepeats+1:end,idx), 'omitnan'),inRes.deltaParams);
    sortedData = arrayfun(@(x,y) sort(abs([x{1}(inRes.normEstRepeats+1:end,idx);y]),'descend'), inRes.deltaParams,contData, 'uni', 0);
    
    % Plot the data in the "scanPlot" structure.
    scanPlot.data = contData./stdData;
    scanPlot.pVals = cell2mat(arrayfun(@(x,y) max([find(x==y{1},1) nan])./nShuffles, abs(contData), sortedData,'uni', 0));

    if i~=3; scanPlot.legendOff = 1; else; scanPlot.legendOff = 0; end 

    plt.scanningBrainEffects(scanPlot);
    xlim([0,6])

    text(0, 4.5, prmTitle{i}, 'FontWeight','bold');
    if i == 1
        scanPlot.colorBar = colorbar;
        scanPlot.colorBar.Position = scanPlot.colorBar.Position.*[0.4,1,1.25,0.8];
        set(scanPlot.colorBar, 'Ticks', [-10, 0 10])
        scanPlot.colorBar.Label.String = '\Delta Value (SDs)';
    end
    ylim([-5.5 5]);
end
end