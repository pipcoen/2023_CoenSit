function RTorTOComparsonWithLMETest(figPanel)
%Load the block if it doesn't exist. Remove mice that have different parameter values (4 mice of 21)
numRef = double(upper(figPanel)-64);

if numRef < 17
    blkDat = spatialAnalysis('all', 'uniscan', 0, 1);
    laserRef = 1;
    siteOrd = {'Frontal'; 'Vis'; 'Lateral'};
    galvoIdx = {[0.6 2; 1.8, 2; 0.6, 3];[1.8 -4; 3,-4; 3,-3];[4.2,-2; 4.2,-3; 4.2,-4]};
else
    blkDat = spatialAnalysis('all', 'biscan', 0, 1);
    siteOrd = {'Frontal'; 'Vis'; 'Parietal'};
    laserRef = 2;
    galvoIdx = {[0.5,2;1.5, 2; 0.5, 3];[1.5 -4; 2.5,-4; 2.5,-3];[1.5,-2; 2.5,-2; 3.5,-2]};
end

if numRef<9; opt.prmType = 'rea';
elseif numRef<17; opt.prmType = 'tim';
elseif numRef<21; opt.prmType = 'slw';
else, opt.prmType = 'res';
end

if ismember(numRef, [1:4 9:12]); opt.contra = 1;
elseif ismember(numRef, [5:8 13:16]); opt.contra = 0;
else, opt.contra = nan;
end

if ismember(numRef, [1,5,9,13,17,21]); opt.trialType = 'visual';
elseif ismember(numRef, [2,6,10,14,18,22]); opt.trialType = 'auditory';
elseif ismember(numRef, [3,7,11,15,19,23]); opt.trialType = 'coherent';
elseif ismember(numRef, [4,8,12,16,20,24]); opt.trialType = 'conflict';
end

%pre-assign performance and reaction structures with nans
nMice = length(blkDat.blks);
clear reacN reacL
[LME.tstVar, LME.stimC, LME.mIdx, LME.lasOn] = deal(cell(length(galvoIdx), 1));
[LMESite.tstVar, LMESite.stimC, LMESite.mIdx, LMESite.siteIdx] = deal([]);
[globalTstVarN, globalTstVarL] = deal(nan*ones(nMice,3));

for mouse = 1:nMice
    iBlk = prc.filtBlock(blkDat.blks(mouse), blkDat.blks(mouse).tri.trialType.(opt.trialType));
    iBlk = prc.filtBlock(iBlk, iBlk.tri.inactivation.galvoPosition(:,2)~=4.5);
    iBlk = prc.filtBlock(iBlk, iBlk.tri.trialType.repeatNum==1 & iBlk.tri.trialType.validTrial & ~iBlk.tri.trialType.blank);

    if numRef > 16
        %Changes specific to the bilateral inactivation data
        iBlk = prc.filtBlock(iBlk, iBlk.tri.stim.visContrast < 0.07);
        iBlk.tri.stim.audDiff(isinf(iBlk.tri.stim.audDiff)) = 0;

        rtLimit = 1.5;
        iBlk.tri.outcome.responseCalc(iBlk.tri.outcome.reactionTime>rtLimit) = nan;
        iBlk.tri.outcome.responseRecorded(iBlk.tri.outcome.reactionTime>rtLimit) = 0;
        iBlk.tri.outcome.reactionTime(iBlk.tri.outcome.reactionTime>rtLimit) = nan;

        rIdx = iBlk.tri.stim.visDiff>0 | (iBlk.tri.stim.visDiff==0 & iBlk.tri.stim.audDiff>0);
        iBlk.tri.outcome.responseCalc(rIdx) = (iBlk.tri.outcome.responseCalc(rIdx)*-1+3).*(iBlk.tri.outcome.responseCalc(rIdx)>0);
        iBlk.tri.stim.audInitialAzimuth(rIdx) = iBlk.tri.stim.audInitialAzimuth(rIdx)*-1;
        iBlk.tri.stim.visInitialAzimuth(rIdx) = iBlk.tri.stim.visInitialAzimuth(rIdx)*-1;
        iBlk.tri.stim.visInitialAzimuth(isinf(iBlk.tri.stim.visInitialAzimuth)) = inf;
        iBlk.tri.stim.conditionLabel(rIdx) = iBlk.tri.stim.conditionLabel(rIdx)*-1;
    else
        iBlk = prc.filtBlock(iBlk, ~ismember(abs(iBlk.tri.inactivation.galvoPosition(:,1)),[0.5; 2; 3.5; 5]) | iBlk.tri.inactivation.laserType==0);
    end

    if strcmpi(opt.prmType, 'rea')
        f2Use = 'reactionTimeComb';
        iBlk.tri.datGlobal = iBlk.tri.outcome.reactionTime;
        iBlk = prc.filtBlock(iBlk, ~isnan(iBlk.tri.outcome.reactionTime));
    elseif strcmpi(opt.prmType, 'tim')
        f2Use = 'fracTimeOutComb';
        iBlk.tri.datGlobal = iBlk.tri.outcome.responseRecorded==0;
    elseif strcmpi(opt.prmType, 'res')
        f2Use = 'fracRightTurnsComb';
        iBlk.tri.datGlobal = iBlk.tri.outcome.responseCalc;
        iBlk = prc.filtBlock(iBlk, ~isnan(iBlk.tri.outcome.responseCalc));
    elseif strcmpi(opt.prmType, 'slw')
        f2Use = 'fracLongResponses';
        iBlk.tri.datGlobal = iBlk.tri.outcome.reactionTime;
        iBlk = prc.filtBlock(iBlk, ~isnan(iBlk.tri.outcome.reactionTime));
    end
    gPosAbs = [abs(iBlk.tri.inactivation.galvoPosition(:,1)) iBlk.tri.inactivation.galvoPosition(:,2)];

    for site = 1:length(galvoIdx)
        gIdx = ismember(gPosAbs,galvoIdx{site}, 'rows');

        fBlk = prc.filtBlock(iBlk, (iBlk.tri.inactivation.laserType==0 | gIdx));

        normBlk = prc.filtBlock(fBlk, fBlk.tri.inactivation.laserType==0);
        lasBlk = prc.filtBlock(fBlk, fBlk.tri.inactivation.laserType==laserRef);

        gPos = lasBlk.tri.inactivation.galvoPosition(:,1);
        aDiff = lasBlk.tri.stim.audDiff;
        vDiff = lasBlk.tri.stim.visDiff;
        if opt.contra == 1
            cIdx = sign(vDiff).*sign(gPos) < 0 | (vDiff==0 & (sign(aDiff).*sign(gPos)<0));
            lasBlk = prc.filtBlock(lasBlk, cIdx);
        elseif opt.contra == 0
            cIdx = sign(vDiff).*sign(gPos) > 0 | (vDiff==0 & (sign(aDiff).*sign(gPos)>0));
            lasBlk = prc.filtBlock(lasBlk, cIdx);
        end

        lasGrds = prc.getGridsFromBlock(lasBlk, 3);
        normGrds = prc.getGridsFromBlock(normBlk, 3);

        globalTstVarN(mouse,site) = mean(normGrds.(f2Use)(~isnan(normGrds.(f2Use))));
        globalTstVarL(mouse,site) = mean(lasGrds.(f2Use)(~isnan(lasGrds.(f2Use))));

        tkIdx = ~isnan(lasGrds.(f2Use));
        lasGrds.visValues(lasGrds.visValues==0) = [-10; 0; 10];
        LME.tstVar{site} = [LME.tstVar{site}; lasGrds.(f2Use)(tkIdx)];
        LME.stimC{site} = [LME.stimC{site}; lasGrds.visValues(tkIdx)];
        LME.mIdx{site} = [LME.mIdx{site}; lasGrds.(f2Use)(tkIdx)*0+mouse];
        LME.lasOn{site} = [LME.lasOn{site}; lasGrds.(f2Use)(tkIdx)*0+1];

        tkIdx = ~isnan(normGrds.(f2Use));
        normGrds.visValues(normGrds.visValues==0) = [-10; 0; 10];
        LME.tstVar{site} = [LME.tstVar{site}; normGrds.(f2Use)(tkIdx)];
        LME.stimC{site} = [LME.stimC{site}; normGrds.visValues(tkIdx)];
        LME.mIdx{site} = [LME.mIdx{site}; normGrds.(f2Use)(tkIdx)*0+mouse];
        LME.lasOn{site} = [LME.lasOn{site}; normGrds.(f2Use)(tkIdx)*0];

        tkIdx = ~isnan(lasGrds.(f2Use));
        LMESite.tstVar = [LMESite.tstVar; lasGrds.(f2Use)(tkIdx)-normGrds.(f2Use)(tkIdx)];
        LMESite.stimC = [LMESite.stimC; lasGrds.visValues(tkIdx)];
        LMESite.mIdx = [LMESite.mIdx; lasGrds.(f2Use)(tkIdx)*0+mouse];
        LMESite.siteIdx = [LMESite.siteIdx; lasGrds.(f2Use)(tkIdx)*0+site];
    end
end
fprintf('Finished collating requested data \n')

%%
if numRef > 16
    LMEtlbs = cellfun(@(w,x,z) table(w,z,x, 'VariableNames',{'tstVar','LaserOn','Mouse'}),...
        LME.tstVar, LME.mIdx, LME.lasOn, 'uni', 0);
    LMEfits = cellfun(@(x) fitlme(x, 'tstVar~LaserOn+(1|Mouse)'), LMEtlbs, 'uni', 0);
else
    LMEtlbs = cellfun(@(w,x,y,z) table(w,z,nominal(x),y, 'VariableNames',{'tstVar','LaserOn','stimC','Mouse'}),...
        LME.tstVar, LME.stimC, LME.mIdx, LME.lasOn, 'uni', 0);
    LMEfits = cellfun(@(x) fitlme(x, 'tstVar~stimC+LaserOn+(1|Mouse)'), LMEtlbs, 'uni', 0);
end
pValLas_Site = cellfun(@(x) x.Coefficients.pValue(contains(x.Coefficients.Name, 'LaserOn')), LMEfits);

compDo = [1 2; 1,3; 2,3];
pValSiteComp = nan*ones(3,1);
for i = 1:size(compDo,1)
    tDat = structfun(@(x) x(ismember(LMESite.siteIdx, compDo(i,:))), LMESite, 'uni', 0);
    if numRef > 16
        tDatTbl = table(tDat.tstVar,tDat.mIdx,tDat.siteIdx,...
            'VariableNames',{'tstVar','Mouse','tSite'});
        tDatFit = fitlme(tDatTbl, 'tstVar~tSite+(1|Mouse)');
    else
        tDatTbl = table(tDat.tstVar,nominal(abs(tDat.stimC)),tDat.mIdx,tDat.siteIdx,...
            'VariableNames',{'tstVar','stimC','Mouse','tSite'});
        tDatFit = fitlme(tDatTbl, 'tstVar~stimC+tSite+(1|Mouse)');
    end
    pValSiteComp(i,1) = tDatFit.Coefficients.pValue(contains(tDatFit.Coefficients.Name, 'tSite'));
end
fprintf('Finished calculating stats using LME models \n')

%%
if strcmpi(opt.prmType, 'rea')
    globalTstVarN = globalTstVarN*1000;
    globalTstVarL = globalTstVarL*1000;
    ylab = '\Delta ReactionTime (ms)';
    axLims = [-35 105];
elseif strcmpi(opt.prmType, 'tim')
    ylab = '\Delta Fraction of timeouts';
    axLims = [-0.08 0.45];
elseif strcmpi(opt.prmType, 'res')
    ylab = '\Delta Fraction rightward choices';
    axLims = [-0.15 0.40];
elseif strcmpi(opt.prmType, 'slw')
    ylab = '\Delta Fraction of slow responses';
    axLims = [-0.05 0.25];
end

%%
figure;
set(gcf, 'position', get(gcf, 'position').*[1 1 0.5 1])
set(gca, 'position', get(gca, 'position').*[1.75 1.3 0.8 1])
hold on

tDatDiff = globalTstVarL - globalTstVarN;
% Threshold differences for plotting;
tDatDiff(tDatDiff>max(axLims)) = max(axLims);
tDatDiff = num2cell(tDatDiff,1);

nXPnts = length(tDatDiff);
yDat = cell2mat(arrayfun(@(x) [tDatDiff{x}; mean(tDatDiff{x})], 1:nXPnts, 'uni', 0));
xDat = 1:nXPnts;

hold on
for i = 1:nXPnts-1
    cellfun(@(y) plot([i+0.1 i+0.9],y,'Color',[0.5,0.5,0.5]), num2cell(yDat(1:end-1,i:i+1),2));
    cellfun(@(y) plot([i+0.1 i+0.9],y,'Color','k','linewidth',1.5), num2cell(yDat(end,i:i+1),2));
end
for i = 1:nXPnts
    if pValLas_Site(i) < 0.05
        txtStr = num2str(pValLas_Site(i), '%.1E');
    else
        txtStr = 'ns';
    end
    text(xDat(1,i), min(axLims)-range(axLims)*0.1, txtStr, ...
        'HorizontalAlignment', 'center', 'fontsize', 8);
end
for i = 1:size(compDo,1)
    if pValSiteComp(i) < 0.05
        txtStr = num2str(pValSiteComp(i), '%.1E');
    else
        txtStr = 'ns';
    end
    xPnts = xDat(compDo(i,:));
    yLev = max(axLims)*(1.05+i*0.05);
    plot(xPnts, yLev*[1 1], 'k')
    text(xPnts(2), yLev, txtStr, ...
        'HorizontalAlignment', 'left', 'fontsize', 8);
end
ylim([axLims(1) yLev]);
xlim([xDat(1,1)-0.5 xDat(1,end)+0.5]);
set(gca, 'XTick', xDat(1,:), 'XTickLabel', siteOrd, 'YTick', [axLims(1) 0 axLims(2)]);
yline(0, '--k', 'Alpha',1)
box off;
ylabel(ylab)
text(xDat(2), min(axLims)-range(axLims)*0.15, 'Inactivated region', 'FontSize', 12, 'HorizontalAlignment', 'center')
