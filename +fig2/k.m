function k
%% This function plots a panel from the manuscript (Figure S5k)
%Set up defaults for the input values. "op2use" is "mean" as default for responses, but changes below depending on the type data being used
s = spatialAnalysis('all', 'variscan', 1, 1);
inactvationSite = 'mos';
trialTypes = {'visual'; 'auditory'};
pltTitle = 'Timed inactivations of frontal cortex';
yRng = [-0.1 0.4];
figure;
%%
binSize = 70;
plotRange = [-100-binSize/2 150+binSize/2];

overlap = binSize-10;
bins = (plotRange(1):(binSize-overlap):plotRange(2)-binSize)';
bins = [bins bins+binSize];
pLim = 10^-3;
xDat = bins(:,1)+binSize/2;

regRef = {'mos', [0.5, 2.0]; 'v1', [2.0, -4.0]; 's1' , [3.5 -0.5]; 'out' , [3 5.5]};

%Set up plotting axes arrangement on figrure
inactvationSite2keep = cell2mat(regRef(contains(regRef(:,1), inactvationSite),2));

fullBlk = prc.filtBlock(s.blks, s.blks.tri.trialType.repeatNum==1 & s.blks.tri.trialType.validTrial);
fullBlk = prc.filtBlock(fullBlk, ismember(abs(fullBlk.tri.inactivation.galvoPosition),abs(inactvationSite2keep), 'rows') | fullBlk.tri.inactivation.laserType==0);
fullBlk = prc.filtBlock(fullBlk, fullBlk.tri.stim.visContrast > 0 | fullBlk.tri.trialType.auditory);

for i = 1:length(trialTypes)
    trialType = trialTypes{i};
    iBlk = prc.filtBlock(fullBlk, fullBlk.tri.trialType.(trialType));

    if strcmpi(trialType(1:3), 'vis');  pltCol = [0.9290, 0.6940, 0.1250]; end
    if strcmpi(trialType(1:3), 'aud');  pltCol = [189, 48, 255]/255; end

    idx2Flip = iBlk.tri.inactivation.galvoPosition(:,1)<0 & iBlk.tri.inactivation.laserType==1;
    iBlk.tri.stim.audDiff(idx2Flip) = -1*iBlk.tri.stim.audDiff(idx2Flip);
    iBlk.tri.stim.visDiff(idx2Flip) = -1*iBlk.tri.stim.visDiff(idx2Flip);
    iBlk.tri.stim.conditionLabel(idx2Flip) = -1*iBlk.tri.stim.conditionLabel(idx2Flip);
    iBlk.tri.inactivation.galvoPosition(idx2Flip,1) = -1*iBlk.tri.inactivation.galvoPosition(idx2Flip,1);
    iBlk.tri.outcome.responseCalc(idx2Flip) = (iBlk.tri.outcome.responseCalc(idx2Flip)*-1+3).*(iBlk.tri.outcome.responseCalc(idx2Flip)>0);

    iBlk = prc.filtBlock(iBlk, ~isnan(iBlk.tri.outcome.responseCalc));
    iBlk.tri.inactivation.data2Use = iBlk.tri.outcome.responseCalc==2;
    iBlk = prc.filtBlock(iBlk, iBlk.exp.numOfTrials>75);
    rIdx = iBlk.tri.stim.visDiff<0 | (iBlk.tri.stim.visDiff==0 & iBlk.tri.stim.audDiff<0);
    iBlk = prc.filtBlock(iBlk, rIdx);

    %Create normBlk and uniBlk which are filtered versions of iBlk with only control or inactivation trials respectively
    normBlk = prc.filtBlock(iBlk, iBlk.tri.inactivation.laserType==0);
    uniBlk = prc.filtBlock(iBlk, iBlk.tri.inactivation.laserType==1);
    laserOffsets = uniBlk.tri.inactivation.laserOnsetDelay*1000;
    sampIdx = arrayfun(@(x,y) laserOffsets>x & laserOffsets<y, bins(:,1), bins(:,2), 'uni', 0);

    nrmDat.fracT = mean(normBlk.tri.inactivation.data2Use);
    nrmDat.nTrue = sum(normBlk.tri.inactivation.data2Use);
    nrmDat.nFalse = sum(~normBlk.tri.inactivation.data2Use);

    lasDat.fracT = cellfun(@(x) mean(uniBlk.tri.inactivation.data2Use(x)), sampIdx);
    lasDat.nTrue = cellfun(@(x) sum(uniBlk.tri.inactivation.data2Use(x)), sampIdx);
    lasDat.nFalse = cellfun(@(x) sum(~uniBlk.tri.inactivation.data2Use(x)), sampIdx);

    CI = 1.96*sqrt((lasDat.fracT.*(1-lasDat.fracT))./(lasDat.nTrue+lasDat.nFalse));
    pltM = lasDat.fracT-nrmDat.fracT;

    plotData = permute(cat(3, pltM, pltM-CI, pltM+CI), [2 1 3]);
    opt.Marker = 'none';
    plt.rowsOfGrid(xDat', plotData, pltCol, opt);

    for i = 1:length(xDat)
        tbl = table([nrmDat.nTrue;lasDat.nTrue(i)],[nrmDat.nFalse;lasDat.nFalse(i)], ...
            'VariableNames',{'right','left'},'RowNames',{'Cnt','Las'});
        [~,pVal(i)] = fishertest(tbl);
    end
    if any(pVal<pLim); plot(xDat(pVal<pLim), max(pltM+CI)+0.05, '.', 'color',  pltCol); end
end
ylim(yRng)
plot(xlim, [0,0], '--k', 'linewidth', 1.5)
plot([0,0], ylim, '--k', 'linewidth', 1.5)
xlabel('Time of inactivation relative to stimulus onset (ms)')
ylabel('\Delta Fraction of rightward choices')
text(-75, 0.25,'Visual trials', 'color', [0.9290, 0.6940, 0.1250], 'FontSize',12, 'FontWeight','bold');
text(-75, 0.22,'Auditory trials', 'color', [189, 48, 255]/255, 'FontSize',12, 'FontWeight','bold');
title(pltTitle);
