function e
%% This function plots a panel from the manuscript (Figure 1e)
blks = spatialAnalysis('all', 'behavior', 0, 1, '');
vis2Use = 0.4; % The visual contrast to use for the plot
blks = blks.blks;

%pre-assign performance and reaction structures with nans
[perf.vis, perf.aud, perf.mul] = deal(nan*ones(length(blks), 1));

% Get indices for example mice
eIdx = contains(arrayfun(@(x) x.exp.subject{1}, blks, 'uni', 0), {'PC022', 'PC051'});
nMice = length(blks);
for i = 1:nMice
    %"normalize" block--removes timeouts, laser, and nan-response trials
    nBlk = spatialAnalysis.getBlockType(blks(i), 'norm');
    % Get "grids" corresponding to differet behavioural features
    grds = prc.getGridsFromBlock(nBlk);
    
    % Get performance on different trial types by subsetting grid
    perf.aud(i) = grds.performance(grds.visValues == 0 & grds.audValues > 0);
    perf.vis(i) = grds.performance(grds.visValues == vis2Use & grds.audValues == 0);
    perf.mul(i) = grds.performance(grds.visValues == vis2Use & grds.audValues > 0);
end
%%
figure;
hold on;
plotData = [perf.aud perf.vis perf.mul]*100;
xDat = (1:size(plotData,2)) - 0.5;
examps = find(eIdx);

for i = 1:size(plotData,2)-1
    plot([xDat(i),xDat(i+1)]+[0.1, -0.1], plotData(~eIdx,i:i+1), ...
        'Color', [0.5,0.5,0.5], 'linewidth', 1);

    plot([xDat(i),xDat(i+1)]+[0.1, -0.1], plotData(examps(1),i:i+1), ...
        '--', 'Color', [0.5,0.5,0.5], 'linewidth', 1);

    plot([xDat(i),xDat(i+1)]+[0.1, -0.1], plotData(examps(2),i:i+1), ...
        ':', 'Color', [0.5,0.5,0.5], 'linewidth', 1);

    meanLine = mean(plotData);
    plot([xDat(i),xDat(i+1)]+[0.1, -0.1], meanLine(i:i+1), 'k', 'linewidth', 2);
end
ylim([50 100])

[~, pVal] = ttest(perf.vis, perf.aud);
pVal = round(pVal, 2, 'significant');
text(0.7, 70, ['A vs V: P<' num2str(pVal)]);

[~, pVal] = ttest(perf.aud, perf.mul);
pVal = round(pVal, 2, 'significant');
text(1.5, 70, ['A vs A=V: P<' num2str(pVal)]);

[~, pVal] = ttest(perf.vis, perf.mul);
pVal = round(pVal, 2, 'significant');
text(1.5, 65, ['V vs A=V: P<' num2str(pVal)]);

xlabel('Stimulus Type')
ylabel('Performance (%)')
set(gca, 'xTick', xDat, 'xTickLabel', {'A', 'V', 'A=V'})
set(gcf, 'position', get(gcf, 'Position').*[1,1,0.7,1])
end