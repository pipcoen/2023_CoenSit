function g
%% This function plots a panel from the manuscript (Figure S2g)
behBlks = spatialAnalysis('all', 'behaviour', 0, 1, '');
behBlks = behBlks.blks;
nMice = length(behBlks);
%%
for i = 1:nMice
    nBlk = spatialAnalysis.getBlockType(behBlks(i), 'norm');
    if     any(isinf(nBlk.tri.stim.audInitialAzimuth)); keyboard; end

    grds = prc.getGridsFromBlock(nBlk, 1);
    grds.fracRightTurns = grds.fracRightTurnsComb;
    
    fracAudL = grds.fracRightTurns(grds.audValues < 0 & grds.visValues==0);
    fracAudR = grds.fracRightTurns(grds.audValues > 0 & grds.visValues==0);
    
    fracVisL = grds.fracRightTurns.*(grds.visValues < 0 & grds.audValues==0);
    fracVisL(fracVisL==0) = nan;
    fracVisR = grds.fracRightTurns.*(grds.visValues > 0 & grds.audValues==0);
    fracVisR(fracVisR==0) = nan;

    closestVL = min(abs((1-fracVisL(:)) - fracAudR(:))) == abs((1-fracVisL) - fracAudR);
    closestVR = min(abs((1-fracVisR(:)) - fracAudL(:))) == abs((1-fracVisR) - fracAudL);
    
    res.audR(i,1) = fracAudR;
    res.audL(i,1) = fracAudL;
    res.visR(i,1) = grds.fracRightTurns(closestVR);
    res.visL(i,1) = grds.fracRightTurns(closestVL);
    res.mulAudR(i,1) = grds.fracRightTurns(grds.audValues > 0 &  grds.visValues == grds.visValues(closestVL));
    res.mulAudL(i,1) = grds.fracRightTurns(grds.audValues < 0 &  grds.visValues == grds.visValues(closestVR));
end
%%
figure;
set(gcf, 'position', get(gcf, 'position').*[1 1 1.3, 1])
subplot(1,2,1);

yDatN = num2cell([res.audR res.mulAudR res.visL],2);
hold on;
cellfun(@(y) plot([1.1 1.9],y(1:2), 'r','HandleVisibility','off'), yDatN);
cellfun(@(y) plot([2.1 2.9],y(2:3), 'r','HandleVisibility','off'), yDatN);
yDatN = cell2mat(yDatN);
plot([1.1 1.9], mean(yDatN(:,1:2)),'k', 'lineWidth', 3);
plot([2.1 2.9], mean(yDatN(:,2:3)),'k', 'lineWidth', 3);
yline(0.5, '--k');

ylabel('Fraction of rightward choices')
set(gca, 'xTick', 1:3, 'xTickLabel', {'A', 'A≠V', 'V'})

title('Auditory right', 'Color','r')
set(gca, 'yTick', [0,0.5,1])

[~, pVal] = ttest(yDatN(:,2)-0.5);
pVal = round(pVal, 2, 'significant');
text(2, 0.1, ['p ~' num2str(pVal)], 'HorizontalAlignment', 'center');

subplot(1,2,2);

yDatN = num2cell([res.audL res.mulAudL res.visR],2);
hold on;
cellfun(@(y) plot([1.1 1.9],y(1:2), 'b','HandleVisibility','off'), yDatN);
cellfun(@(y) plot([2.1 2.9],y(2:3), 'b','HandleVisibility','off'), yDatN);

yDatN = cell2mat(yDatN);
plot([1.1 1.9], mean(yDatN(:,1:2)),'k', 'lineWidth', 3);
plot([2.1 2.9], mean(yDatN(:,2:3)),'k', 'lineWidth', 3);
yline(0.5, '--k');

set(gca, 'xTick', 1:3, 'xTickLabel', {'A', 'A≠V', 'V'},'ycolor', 'none')

title('Auditory left', 'Color','b')
set(gca, 'yTick', [0,0.5,1])

[~, pVal] = ttest(yDatN(:,2)-0.5);
pVal = round(pVal, 2, 'significant');
text(2, 0.1, ['p ~' num2str(pVal)], 'HorizontalAlignment', 'center');
end