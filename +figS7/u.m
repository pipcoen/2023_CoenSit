function u
%% This function plots a panel from the manuscript (Figure S7u)
sAct = spatialAnalysis('all', 'm2ephysmod',1,1,'eph','multiSpaceWorld');
sAct = sAct.blks;
sPass = spatialAnalysis('all', 'm2ephysmod',1,1,'eph','multiSpaceWorldPassive');
sPass = sPass.blks;

% sAct = prc.filtBlock(sAct, sAct.tri.trialType.validTrial & ~isnan(sAct.tri.outcome.responseCalc));
sAct = prc.filtBlock(sAct, sAct.tri.trialType.validTrial & ~sAct.tri.trialType.blank);
sPass = prc.filtBlock(sPass, ~sPass.tri.trialClass.closedLoopOnsetTone ...
    & ~sPass.tri.trialClass.rewardClick...
    & ~sPass.tri.trialClass.blank);
%%
wheelTVInterp = cell(2,1);
for j = 1:2
    if j == 1
        wheelTV = sAct.tri.timeline.wheelTraceTimeValue;
        trialStEn = round(sAct.tri.timings.trialStartEnd*200)/200;
        wheelTVInterp{j,1} = cell(sAct.tot.trials,1);
    else
        wheelTV = sPass.tri.timeline.wheelTraceTimeValue;
        trialStEn = round(sPass.tri.timings.trialStartEnd*200)/200;
        wheelTVInterp{j,1} = cell(sPass.tot.trials,1);
    end
    wheelTV(cellfun(@(x) size(x,1)==1, wheelTV)) = deal({[-0.5,0;0.5,0]});
    wheelTV = cellfun(@(x) [double(x(:,1))+(rand(length(x(:,1)),1)/1e5) double(360*x(:,2)/(4*360))], wheelTV, 'uni', 0);
    wheelTVInterp{j,1} = cellfun(@(x,y) [(y(1):0.005:y(2))', ...
        interp1(x(:,1), x(:,2), y(1):0.005:y(2), 'nearest','extrap')'], wheelTV, num2cell(trialStEn,2), 'uni', 0);
end
%%
wheelPosPlot = cell(1,2);
segVect = -0.05:0.005:0.5;
zeroPnt = find(segVect==0);
evalPnt = find(segVect==0.5);

for j = 1:2
    if j == 1
        stimTime = double(min([sAct.tri.timeline.visStimPeriodOnOff(:,1), sAct.tri.timeline.audStimPeriodOnOff(:,1)],[],2));
    else
        stimTime = double(min([sPass.tri.timeline.visStimPeriodOnOff(:,1), sPass.tri.timeline.audStimPeriodOnOff(:,1)],[],2));
    end
    stimTime = round(stimTime*200)/200;
    wheelPosPlot{1,j} = cellfun(@(x,y) [round((x(:,1)-y)*200)/200, x(:,2)],  wheelTVInterp{j}, num2cell(stimTime+segVect(1)), 'uni', 0);
    zIdx = cellfun(@(x) find(x(:,1)==0)+[0:(length(segVect)-1)], wheelPosPlot{1,j}, 'uni', 0);
    wheelPosPlot{1,j} = cell2mat(cellfun(@(x,y) x(y,2)',  wheelPosPlot{1,j}, zIdx, 'uni', 0));
    wheelPosPlot{1,j} = abs(bsxfun(@minus, wheelPosPlot{1,j}, wheelPosPlot{1,j}(:,zeroPnt)));
end
%%
figure;
hold on
for i = 1:2
    mDat = nan*ones(sAct.tot.subjects, length(segVect));
    hold on
    if i == 1
        subIdx = sAct.tri.subjectRef;
        cCol = [1,0,0];
    else
        subIdx = sPass.tri.subjectRef;
        cCol = [0,0,1];
    end
    for j = 1:sAct.tot.subjects
        mDat(j,:) = mean(wheelPosPlot{1,i}(subIdx==j,:),'omitnan');
    end
    opt.Marker = 'none';
    opt.lineWidth = 0.5;
    plt.rowsOfGrid(segVect, mDat, repmat(cCol,5,1), opt)
    plot(segVect, mean(mDat), 'color', cCol, 'linewidth', 3);
end
xline(0, '--k', 'LineWidth',2, 'alpha', 1)
xlim([-0.05, 0.5]);
box off;
xlabel('Time from stimulus onset')
ylabel('Wheel position (deg)')
text(0.2, 20, 'Passive presentation', 'color', 'b')
text(0.2, 22, 'Active behaviour', 'color', 'r')