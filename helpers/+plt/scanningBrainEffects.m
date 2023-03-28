function scanningBrainEffects(scanPlot)
if ~isfield(scanPlot, 'addTrialNumber'); scanPlot.addTrialNumber = 0; end
if ~isfield(scanPlot, 'pVals'); scanPlot.pVals = 0; end
if ~isfield(scanPlot, 'sigLevels') || isempty(scanPlot.sigLevels); scanPlot.sigLevels = [0.01; 0.001; 0.0001]; end
if ~isfield(scanPlot, 'legendOff') || isempty(scanPlot.legendOff); scanPlot.legendOff = 0; end
plt.allenOutline;
hold on;
plotData = scanPlot.data(:);
MLCoord = scanPlot.gridXY{1}(:);
APCoord = scanPlot.gridXY{2}(:);
pVals = scanPlot.pVals(:);
minSig = min(pVals);
if ~any(scanPlot.pVals(:)) || minSig > max(scanPlot.sigLevels)
    warning(['Pvals do not exist or are all > ' num2str(max(scanPlot.sigLevels))]);
    spotSize = APCoord*0+(200./(length(scanPlot.sigLevels)*1.5))*0.2;
else
    spotSize = sum(cell2mat(arrayfun(@(x) (pVals<x), scanPlot.sigLevels, 'uni', 0)'),2);
    legendRef = scanPlot.sigLevels(scanPlot.sigLevels>minSig);
    lengendSizes = (1:sum(scanPlot.sigLevels>minSig))'.*(200./(length(scanPlot.sigLevels)*1.5));
    spotSize(spotSize==0) = 0.2;
    spotSize = spotSize*(200./(length(scanPlot.sigLevels)*1.5));
    legendRef = [legendRef, legendRef*0+4, legendRef*0+(3+length(legendRef))-(0:length(legendRef)-1)' lengendSizes];
end
sigIdx = pVals<=max(scanPlot.sigLevels);
h1 = scatter(MLCoord(sigIdx), APCoord(sigIdx), spotSize(sigIdx), plotData(sigIdx), 'o', 'filled'); axis equal; drawnow
set(h1, 'MarkerEdgeColor', 'k');
scatter(MLCoord(~sigIdx), APCoord(~sigIdx), spotSize(~sigIdx), plotData(~sigIdx), 'o', 'filled'); axis equal; drawnow
if exist('legendRef', 'var') && ~scanPlot.legendOff
    scatter(legendRef(:,2), legendRef(:,3)-1, legendRef(:,4), 'k', 'o', 'filled'); axis equal; drawnow
    arrayfun(@(x,y,z) text(x,y, num2str(z), 'VerticalAlignment', 'middle'), legendRef(:,2)+0.5, legendRef(:,3)-1, legendRef(:,1))
end

xlim([-5.5 6])
ylim([-5.5 6])
box off; set(gca, 'ycolor', 'w', 'xcolor', 'w', 'xTick', -5:1:5, 'yTick', -5:4, 'gridAlpha', 0.75, 'gridlinestyle', ':', 'GridColor', 'k', 'LineWidth', 1);
colormap(plt.redBlueMap(64));
if isfield(scanPlot, 'colorBarLimits'); caxis(scanPlot.colorBarLimits); else, caxis([-0.6 0.6]); end
end