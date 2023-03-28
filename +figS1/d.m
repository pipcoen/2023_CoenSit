function d
%% This function plots a panel from the manuscript (Figure S1d)
rawBlk = load([prc.pathFinder('processeddirectory') '\XSupData\2019-03-22_1_PC043_Block']);
rawBlk = rawBlk.block;
s = spatialAnalysis('PC043', '2019-03-22', 0, 1, 'raw');
%%
sR = 1000;

wheelTime = 0:1/sR:rawBlk.inputs.wheelTimes(end);
% Converts the numbers from the rotary encoder to degrees of wheel movement
wheelDeg = 360*rawBlk.inputs.wheelValues/(4*360);
wheelDeg = interp1(rawBlk.inputs.wheelTimes, wheelDeg, wheelTime, 'pchip', 'extrap');

segTime = 368:1/sR:390;
wheelDegSeg = interp1(wheelTime, wheelDeg, segTime, 'nearest', 'extrap')';

idx = find(s.blks.tri.timings.stimPeriodStart>segTime(1) & s.blks.tri.timings.stimPeriodStart < segTime(end));
reactTimes = s.blks.tri.timings.stimPeriodStart(idx) + s.blks.tri.outcome.reactionTime(idx) - segTime(1);
feedBackTimes = s.blks.tri.timings.stimPeriodStart(idx) + s.blks.tri.outcome.timeToFeedback(idx) - segTime(1);
decThrTimes = s.blks.tri.timings.stimPeriodStart(idx) + s.blks.tri.outcome.timeToResponseThresh(idx) - segTime(1);
stimTimes = s.blks.tri.timings.stimPeriodStart(idx) - segTime(1);

moveDir = s.blks.tri.outcome.responseCalc(idx);
segTime = segTime-segTime(1);
wheelDegSeg = (wheelDegSeg-wheelDegSeg(1))*-1;

figure
plot(segTime', wheelDegSeg, 'k');
hold on;
box off;
yL = 185;
set(gca, 'YLim', [-150 150], 'XLim', [0 22], 'YTick', [-150 150])
arrayfun(@(x) patch(x+[0 0 0.5 0.5 0], yL*[-1 1 1 -1 -1], 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none'), stimTimes);

for i = 1:length(reactTimes)
    moveIdx = (round(reactTimes(i)*sR):round(feedBackTimes(i)*sR))+1;
    if moveDir(i) == 2; lCol = 'r'; else, lCol = 'b'; end
    plot(segTime(moveIdx), wheelDegSeg(moveIdx), lCol);
end
plot(decThrTimes, wheelDegSeg(round(decThrTimes*sR)), 'sk', 'MarkerSize', 6, 'MarkerFaceColor', 'k');
xlabel('Time (s)')
ylabel('Wheel position (deg)')
end