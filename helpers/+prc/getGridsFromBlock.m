function grids = getGridsFromBlock(blk, fastMode)
%% Function to extract useful "grids" from a block, arranged with aud/vis values in rows/columns

if ~exist('fastMode', 'var'); fastMode = 0; end
outC = blk.tri.outcome;
responseCorrect = blk.tri.stim.correctResponse==outC.responseCalc;

grids = prc.makeGrid(blk);                                                       %Grids of aud/vis values and the condition labels
grids.performance = prc.makeGrid(blk, responseCorrect==1, @mean,'abscondition'); %Grid of performance
grids.numTrials = prc.makeGrid(blk, outC.responseCalc==1, @length);              %Grid of trial numbers

%Get grid of fraction of right turns for each session, then calculate the SE across sessions. Mean of these sessions is the overall fration
if fastMode < 2
    grids.fracRightTurns = prc.makeGrid(blk, outC.responseCalc==2, @nanmean, [], 1);
    grids.fracRightTurnsSE = nanstd(grids.fracRightTurns,[],3)./sqrt(size(grids.fracRightTurns,3));
    grids.fracRightTurns = nanmean(grids.fracRightTurns, 3);
    
    %Get confidence binomial intervals for fraction of right turns (more appropriate for combined mice)
    numRightTurns = prc.makeGrid(blk, blk.tri.outcome.responseCalc==2, @sum);
    [~,confInterval] = arrayfun(@(x,z) binofit(x, z, 0.05), numRightTurns, grids.numTrials, 'uni', 0);
    grids.fracRightTurnsLowBound = cell2mat(cellfun(@(x) permute(x(:,1), [3,2,1]), confInterval, 'uni', 0)).*(grids.fracRightTurns*0+1);
    grids.fracRightTurnsHighBound = cell2mat(cellfun(@(x) permute(x(:,2), [3,2,1]), confInterval, 'uni', 0)).*(grids.fracRightTurns*0+1);
end

%Get grid of meadian reaction times for each session, then calculate the SE across sessions. Mean of these sessions is the overall reaction time
if fastMode < 3
    grids.reactionTime = prc.makeGrid(blk, outC.reactionTime, @nanmedian, [], 1);
    grids.reactionTimeSE = nanstd(grids.reactionTime,[],3)./sqrt(size(grids.reactionTime,3));
    grids.reactionTime = nanmean(grids.reactionTime, 3);
end

grids.fracRightTurnsComb = prc.makeGrid(blk, outC.responseCalc==2, @nanmean);
grids.reactionTimeComb = prc.makeGrid(blk, outC.reactionTime, @nanmedian);
grids.timeToResponseThreshComb = prc.makeGrid(blk, outC.timeToResponseThresh, @nanmedian);
grids.fracLongResponses = prc.makeGrid(blk, outC.reactionTime>0.3, @nanmean);
grids.fracTimeOutComb = prc.makeGrid(blk, outC.responseRecorded==0, @nanmean);


if isfield(blk.tri, 'timeline') && all(strcmp('ephys', blk.exp.expType))
    outC = blk.tri.timeline;
    minStimTime = min([outC.audStimPeriodOnOff(:,1), outC.visStimPeriodOnOff(:,1)], [], 2);
    outC.reactionTime = outC.choiceInitTimeDir(:,1)-minStimTime;
    outC.timeToResponseThresh = outC.choiceThreshTimeDir(:,1)-minStimTime;
    
    grids.reactionTimeTL = prc.makeGrid(blk, outC.reactionTime, @nanmedian, [], 1);
    grids.reactionTimeTLSE = nanstd(grids.reactionTime,[],3)./sqrt(size(grids.reactionTime,3));
    grids.reactionTimeTL = nanmean(grids.reactionTime, 3);
    grids.reactionTimeCombTL = prc.makeGrid(blk, outC.reactionTime, @nanmedian);
    grids.timeToResponseThreshCombTL = prc.makeGrid(blk, outC.timeToResponseThresh, @nanmedian);
end

end