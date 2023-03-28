function l
%% This function plots a panel from the manuscript (Figure S5l)
opt.prmType = 'tim';
opt.contra = 1;
opt.trialType = {'coherent';'conflict'};

blkDat = spatialAnalysis('all', 'uniscan', 0, 1);
laserRef = 1;
siteOrd = {'V-Coh'; 'V-Con'; 'F-Coh'; 'F-Con'};
galvoIdx = {[1.8 -4; 3,-4; 3,-3],[1.8 -4; 3,-4; 3,-3],[0.6 2; 1.8, 2; 0.6, 3],[0.6 2; 1.8, 2; 0.6, 3]};

%pre-assign performance and reaction structures with nans
nMice = length(blkDat.blks);
clear reacN reacL
[LME.tstVar, LME.stimC, LME.mIdx, LME.lasOn] = deal(cell(length(galvoIdx), 1));
[LMESite.tstVar, LMESite.stimC, LMESite.mIdx, LMESite.siteIdx] = deal([]);
[globalTstVarN, globalTstVarL] = deal(nan*ones(nMice,3));

for mouse = 1:nMice
    iBlk = prc.filtBlock(blkDat.blks(mouse), blkDat.blks(mouse).tri.inactivation.galvoPosition(:,2)~=4.5);
    iBlk = prc.filtBlock(iBlk, iBlk.tri.trialType.repeatNum==1 & iBlk.tri.trialType.validTrial & ~iBlk.tri.trialType.blank);
    iBlk = prc.filtBlock(iBlk, ~ismember(abs(iBlk.tri.inactivation.galvoPosition(:,1)),[0.5; 2; 3.5; 5]) | iBlk.tri.inactivation.laserType==0);

    if strcmpi(opt.prmType, 'rea')
        f2Use = 'reactionTimeComb';
        iBlk.tri.datGlobal = iBlk.tri.outcome.reactionTime;
        iBlk = prc.filtBlock(iBlk, ~isnan(iBlk.tri.outcome.reactionTime));
    elseif strcmpi(opt.prmType, 'tim')
        f2Use = 'fracTimeOutComb';
        iBlk.tri.datGlobal = iBlk.tri.outcome.responseRecorded==0;
    end
    gPosAbs = [abs(iBlk.tri.inactivation.galvoPosition(:,1)) iBlk.tri.inactivation.galvoPosition(:,2)];

    for site = 1:length(galvoIdx)
        gIdx = ismember(gPosAbs,galvoIdx{site}, 'rows');

        fBlk = prc.filtBlock(iBlk, (iBlk.tri.inactivation.laserType==0 | gIdx));
        if mod(site,2) == 0
            fBlk = prc.filtBlock(fBlk, fBlk.tri.trialType.conflict);
        else
            fBlk = prc.filtBlock(fBlk, fBlk.tri.trialType.coherent);
        end

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
LMEtlbs = cellfun(@(w,x,y,z) table(w,z,nominal(x),y, 'VariableNames',{'tstVar','LaserOn','stimC','Mouse'}),...
    LME.tstVar, LME.stimC, LME.mIdx, LME.lasOn, 'uni', 0);
LMEfits = cellfun(@(x) fitlme(x, 'tstVar~stimC+LaserOn+(1|Mouse)'), LMEtlbs, 'uni', 0);
pValLas_Site = cellfun(@(x) x.Coefficients.pValue(contains(x.Coefficients.Name, 'LaserOn')), LMEfits);

compDo = [1 3; 2,4];
pValSiteComp = nan*ones(3,1);
for i = 1:size(compDo,1)
    tDat = structfun(@(x) x(ismember(LMESite.siteIdx, compDo(i,:))), LMESite, 'uni', 0);
        tDatTbl = table(tDat.tstVar,nominal(abs(tDat.stimC)),tDat.mIdx,tDat.siteIdx,...
            'VariableNames',{'tstVar','stimC','Mouse','tSite'});
        tDatFit = fitlme(tDatTbl, 'tstVar~stimC+tSite+(1|Mouse)');
    pValSiteComp(i,1) = tDatFit.Coefficients.pValue(contains(tDatFit.Coefficients.Name, 'tSite'));
end
fprintf('Finished calculating stats using LME models \n')

%%
if strcmpi(opt.prmType, 'rea')
    globalTstVarN = globalTstVarN*1000;
    globalTstVarL = globalTstVarL*1000;
    ylab = '\Delta ReactionTime (ms)';
    axLims = [-35 100];
elseif strcmpi(opt.prmType, 'tim')
    ylab = '\Delta Fraction of timeouts';
    axLims = [-0.08 0.45];
end

%%
figure;
set(gcf, 'position', get(gcf, 'position').*[1 1 0.6 1])
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
for i = 1:2:nXPnts-1
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
