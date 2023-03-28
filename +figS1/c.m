function c
%% This function plots a panel from the manuscript (Figure S1c)
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

% Find the date when stage 4 was reached (3x stage 4 sessions)
stage4Reached = arrayfun(@(x) min(strfind(expStage(x,:)==4, [1 1 1])), 1:numMice)';
stage4Fraction = expStage*0;

% Plot the fraction of mice at the final stage for each session
figure;
for i = 1:numMice; stage4Fraction(i, stage4Reached(i):end) = 1; end
plot(1:length(stage4Fraction), mean(stage4Fraction), 'k', 'linewidth', 2)
xlim([1 30]);
box off;

ylabel('Fraction of mice at stage 4')
xlabel('Training sessions')
set(gcf, 'position', get(gcf, 'Position').*[1,1,0.5,1])
end