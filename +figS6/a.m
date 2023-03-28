function a
%% This function plots a panel from the manuscript (Figure S6a)
s = spatialAnalysis('all', 'm2ephysgood', 1, 1);
allenPath = [prc.pathFinder('processeddirectory') 'XSupData\Atlas\\allenCCF\'];
tv = readNPY([allenPath 'template_volume_10um.npy']); % grey-scale "background signal intensity"
av = readNPY([allenPath 'annotation_volume_10um_by_index.npy']); % the number at each pixel labels the area, see note below
st = loadStructureTree([allenPath 'structure_tree_safe_2017.csv']); % a table of what all the labels mean

%%
atlas.tv = tv;
atlas.st = st;
atlas.av = av;
s.blks  = kil.getClusterLoactionsInAllenSpace(s.blks, [], atlas);

probeLength = 3840;
[probeLines, pIdx, probeRef] = unique(s.blks.pen.calcLine, 'rows');
probeCount = groupcounts(probeRef);
probeTips = s.blks.pen.calcTip(pIdx,:);
scalingFactor = s.blks.pen.scalingFactor(pIdx);
colC = jet(5);

figure;
subplot(2,2,1:2)
axH = gca;
hold on
kil.plotAllenOutlines('sag', {'MOs'}, axH, tv, st, av)
for i = 1:length(pIdx)
    startPoint = probeTips(i,:)-(probeLength/10).*probeLines(i,:)*scalingFactor(i);
    probeStEn = [startPoint' probeTips(i,:)'];
    plot(axH, probeStEn(3,:), probeStEn(2,:), 'color', colC(probeCount(i),:), 'linewidth', 1.5)
end
colormap(jet(5))
colorbar

subplot(2,2,3)
axH = gca;
hold on
kil.plotAllenOutlines('cor', {'MOs'}, axH, tv, st, av)
for i = 1:length(pIdx)
    startPoint = probeTips(i,:)-(probeLength/10).*probeLines(i,:)*scalingFactor(i);
    probeStEn = [startPoint' probeTips(i,:)'];
    plot(axH, probeStEn(1,:), probeStEn(2,:), 'color', colC(probeCount(i),:), 'linewidth', 1.5)
end

subplot(2,2,4)
axH = gca;
hold on
kil.plotAllenOutlines('top', {'MOs'}, axH, tv, st, av)
for i = 1:length(pIdx)
    startPoint = probeTips(i,:)-(probeLength/10).*probeLines(i,:)*scalingFactor(i);
    probeStEn = [startPoint' probeTips(i,:)'];
    plot(axH, probeStEn(1,:), probeStEn(3,:), 'color', colC(probeCount(i),:), 'linewidth', 1.5)
end

plt.tightSubplot(nRows,nCols,4,axesGap,botTopMarg,lftRgtMarg); cla;
colorLabels = repmat([0.5 0.5 0.5],s.blks.tot.clusters,1);
areaIdx = contains(s.blks.clu.parent, {'MOs'; 'FRP'});
colorLabels(areaIdx,:) = repmat([0 0 0], sum(areaIdx), 1);
plt.clustersInBrain(s.blks.clu, colorLabels,1);
%%
lfpEx = 28;
plt.tightSubplot(nRows,nCols,5,axesGap,botTopMarg,lftRgtMarg); cla;
powerSpectra = flipud(log(s.blks.pen.lfpPowerSpectra{lfpEx}.powerSpectra(1:100,:)'));
freqPoints = s.blks.pen.lfpPowerSpectra{lfpEx}.freqPoints(1:100);

refChannels = [5; 44; 81; 120; 157; 196; 233; 272; 309; 348];
chan2Plot = setdiff(1:384, refChannels);

powerSpectra = bsxfun(@minus, powerSpectra, median(powerSpectra));
imagesc(freqPoints,1:length(chan2Plot), powerSpectra(chan2Plot,:));
colormap default
colorbar;
end