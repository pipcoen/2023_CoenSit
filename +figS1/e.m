function e
%% This function plots a panel from the manuscript (Figure S1e)
rawBlk = load([prc.pathFinder('processeddirectory') '\XSupData\2019-03-22_1_PC043_Block']);
rawBlk = rawBlk.block;
s = spatialAnalysis('PC043', '2019-03-22', 0, 1, 'raw');

sR = 1000;
% Velocity threshold is 20% of the movement required for a decision
velThresh = s.blks(1).exp.wheelTicksToDecision{1}*0.2; 

wheelTime = 0:1/sR:rawBlk.inputs.wheelTimes(end);
% Converts the numbers from the rotary encoder to degrees of wheel movement
wheelDeg = 360*rawBlk.inputs.wheelValues/(4*360);
wheelDeg = interp1(rawBlk.inputs.wheelTimes, wheelDeg, wheelTime, 'pchip', 'extrap');

figure
hold on;
blk = s.blks;
blk = prc.filtBlock(blk, ~isnan(blk.tri.outcome.reactionTime));
timeWindow = -0.05:0.001:0.1;

for i=1:blk.tot.trials
    timeRef = timeWindow + blk.tri.outcome.reactionTime(i) +  blk.tri.timings.stimPeriodStart(i);
    wheelPosSeg = interp1(wheelTime, wheelDeg, timeRef, 'nearest', 'extrap')';
    wheelPosPreSeg = interp1(wheelTime, wheelDeg, timeRef-0.01, 'nearest', 'extrap')';
    wheelVelSeg = (wheelPosSeg-wheelPosPreSeg)*100;
    if blk.tri.outcome.responseCalc(i) == 2; lCol = [1,0,0,0.075]; else, lCol =  [0,0,1,0.075]; end
    plot(timeWindow, wheelVelSeg, 'color', lCol);
end
plot(0,0,'ok', 'MarkerFaceColor','w')
xlim([timeWindow(1) timeWindow(end)])

ylim([-370 370])
set(gca, 'YTick', [-370 0 370]);
box off;
plot(xlim, velThresh*[1 1], '--k');
plot(xlim, velThresh*[1 1]*-1, '--k');
xlabel('Time from movement onset (s)')
ylabel('Wheel velocity (deg/s)')
end