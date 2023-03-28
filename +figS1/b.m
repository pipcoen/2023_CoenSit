function b
%% This function plots a panel from the manuscript (Figure S1b)
lerBlks = spatialAnalysis('all', 'learning', 0, 0, '');

% Determine the training stage of each session based on the stimulus types
numMice = length(lerBlks.blks);
numExp = 35; %hardcoded in prc.keyDates;
expPer = arrayfun(@(x) permute(cell2mat(x.exp.performanceAVM),[3 1 2]), lerBlks.blks, 'uni', 0);
performanceAVM = permute(cell2mat(expPer), [3 2 1]);

stage4Exps = nan*ones(numMice, numExp);
conflictsPresent = arrayfun(@(x) unique(x.tri.expRef(x.tri.trialType.conflict)), lerBlks.blks, 'uni', 0);
for i = 1:numMice; stage4Exps(i, conflictsPresent{i}) = 1; end
expStage = squeeze(sum(~isnan(performanceAVM),1))' + ~isnan(stage4Exps);

figure;
selectedMouse = 10;
for i = 1:4
    subplot(1,4,i);
    recType = {'last'; 'last'; 'last'; 'first'};
    expIdx = find(expStage(selectedMouse, :)==i, 2, recType{i});
    expIdx = ismember(1:length(lerBlks.blks(selectedMouse).exp.subject), expIdx);
    blk = prc.filtBlock(lerBlks.blks(selectedMouse), expIdx, 'exp');
    [~, ~, paramSets] = prc.uniquecell(blk.exp.conditionParametersAV);
    blk = prc.filtBlock(blk, paramSets==mode(paramSets), 'exp');
    blk = prc.filtBlock(blk, ~isnan(blk.tri.outcome.responseCalc));
    grids = prc.getGridsFromBlock(blk);
    plt.rowsOfGrid(grids.visValues(1,:), grids.fracRightTurns, plt.selectRedBlueColors(grids.audValues(:,1)));
    
    xlim([-0.8 0.8]);
    ylim([0 1]);
    xL = xlim; hold on; plot(xL,[0.5 0.5], '--k', 'linewidth', 1.5);
    yL = ylim; hold on; plot([0 0], yL, '--k', 'linewidth', 1.5);
    box off;
    axis square
    ylim([0 1]);
    
    if i ~= 1
        set(gca, 'ycolor', 'none')
    else
        ylabel('p(Rightward)')
    end
    title(['Stage ' num2str(i)])
    xlabel('Visual contrast (%)')
    set(gca, 'XTick', [-0.8 0 0.8], 'yTick', [0 1])
end
set(gcf, 'position', get(gcf, 'Position').*[1,1,2,0.7])
end