function m
%% This function plots a panel from the manuscript (Figure S1m)
behBlks = spatialAnalysis('all', 'behaviour', 0, 1, '');
behBlks = behBlks.blks;
nMice = length(behBlks);
reacT = deal(nan*ones(3, 9, nMice));
visRef = [-1*[0.8, 0.4, 0.2, 0.1] 0 0.1, 0.2, 0.4, 0.8];
%%
for i = 1:nMice
    % 'norm' removes repeats, timeouts, and opto trials 
    nBlk = spatialAnalysis.getBlockType(behBlks(i), 'norm');
    % Filter out trials with contrasts difference to 'visRef'--because some
    % mice had additional contrasts
    nBlk = prc.filtBlock(nBlk, ismember(nBlk.tri.stim.visDiff, visRef));

    % Update the list of conditionParametersAV to include the trial
    % contrasts specified above
    keepIdx = ismember(nBlk.exp.conditionParametersAV{1}(:,2), visRef);
    nBlk.exp.conditionParametersAV = cellfun(@(x) x(keepIdx,:), nBlk.exp.conditionParametersAV, 'uni', 0);
    nBlk.exp.conditionLabels = cellfun(@(x) x(keepIdx,:), nBlk.exp.conditionLabels, 'uni', 0);

    % Get "grids" corresponding to differet behavioural features
    grds = prc.getGridsFromBlock(nBlk);
    
    % Extract information for each mouse
    reacT(:,:,i) = grds.reactionTime;         % reaction times
end

%%
figure
ylim([180 260])
xlim([-80 80])
set(gca, 'XTick', [-80 -40 0 40 80], 'YTick', [180,260]);
meanData = mean(reacT,3);
seData = std(reacT,[],3)./sqrt(nMice);
plotData = cat(3, meanData, meanData-seData, meanData+seData);
plt.rowsOfGrid(visRef(1,:)*100, plotData*1000, plt.selectRedBlueColors([-60 0 60]));
axis square
ylabel('Reaction time (ms)')
xlabel('Visual contrast (%)')
end