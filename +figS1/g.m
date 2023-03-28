function g
%% This function plots a panel from the manuscript (Figure S1g)
svmMod = load([prc.pathFinder('processeddirectory') '\XSupData\figS1gSVMModelData']);
svmMod =  svmMod.svmMod;

figure;
opt.Marker = 'none';
cCol = [[255, 215, 0]/255; 1 0 1];
%%
for i = 1:2
    modelPerf = cell2mat(cellfun(@(x) mean(x(:,:,i))', svmMod.modPerf, 'uni', 0))';
    meanData = mean(modelPerf);
    seData = std(modelPerf)./sqrt(size(modelPerf,1));
    plotData = cat(3, meanData, meanData-seData, meanData+seData);
    plt.rowsOfGrid(svmMod.svmTimes, plotData, cCol(i,:), opt);
end
xlim([0.05 0.3]);
ylim([0.49 0.8])
set(gca, 'XTick', [0.05 0.3], 'XTickLabel', [50,300])
xlabel('Peri-stimulus time (ms)')
ylabel('Prediction accuracy (%)')
end