function b
%% This function plots a panel from the manuscript (Figure 1b)
blks = spatialAnalysis('all', 'behavior', 0, 1, '');
vis2Use = 0.4; % The visual contrast to use for the plot
blks = blks.blks;

%pre-assign performance and reaction structures with nans
[reac.vis, reac.aud, reac.coh, reac.con] = deal(nan*ones(length(blks), 1));
allRTs = cell(length(blks), 1);

% Get indices for example mice
eIdx = contains(arrayfun(@(x) x.exp.subject{1}, blks, 'uni', 0), {'PC022', 'PC051'});
nMice = length(blks);
for i = 1:nMice
    %"normalize" block--removes timeouts, laser, and nan-response trials
    nBlk = spatialAnalysis.getBlockType(blks(i), 'norm');
    % Get "grids" corresponding to differet behavioural features
    grds = prc.getGridsFromBlock(nBlk);
    
    % Get reaction times on different trial types by subsetting grid
    reac.aud(i) = mean(grds.reactionTime(grds.visValues == 0 & grds.audValues ~= 0))*1000;
    reac.vis(i) = mean(grds.reactionTime(abs(grds.visValues) == vis2Use & grds.audValues == 0))*1000;

    % Combine all reaction times
    allRTs{i,1} = nBlk.tri.outcome.reactionTime;
    
    % Index coherent and conflict trials and get reaction times
    cohIdx = grds.visValues.*grds.audValues > 0 &~isnan(grds.reactionTime) & abs(grds.visValues) == vis2Use;
    conIdx = grds.visValues.*grds.audValues < 0 &~isnan(grds.reactionTime) & abs(grds.visValues) == vis2Use;
    reac.coh(i) = mean(grds.reactionTime(cohIdx))*1000;
    reac.con(i) = mean(grds.reactionTime(conIdx))*1000;
end
% Calculate the "offset" which is the mean reaction time across trial types
%%
figure;
hold on;
offset = mean([reac.aud reac.vis reac.coh reac.con],2);
plotData = bsxfun(@minus, [reac.aud reac.vis reac.coh reac.con], offset);
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

ylim([-35 35])

[~, pVal] = ttest(reac.vis, reac.aud);
pVal = round(pVal, 2, 'significant');
text(0, 35, ['A vs V: P<' num2str(pVal)]);

[~, pVal] = ttest(reac.coh, reac.con);
pVal = round(pVal, 2, 'significant');
text(2, 35, ['A=V vs A≠V: P<' num2str(pVal)]);

xlabel('Stimulus type')
ylabel('Relative reaction time (ms)')
set(gca, 'xTick', xDat, 'xTickLabel', {'A', 'V', 'A=V', 'A≠V'})
set(gcf, 'position', get(gcf, 'Position').*[1,1,0.7,1])
end