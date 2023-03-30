function h
%% This function plots a panel from the manuscript (Figure 2h)
testSite = 'MOs';
uniMice = {'PC027'; 'PC029'; 'DJ008'; 'DJ006'; 'DJ007'};
logLikContTest = nan*(ones(length(uniMice),2));
for i = 1:length(uniMice)
    uniBlks = spatialAnalysis(uniMice{i}, 'uniscan',1,1);
    mOpt = struct;
    mOpt.contOnly = 1;
    mOpt.nRepeats = 1;
    mOpt.useDif = 0;
    cGLM = uniBlks.getModelFitsToInactivationData(mOpt);
    
    mOpt.contParams = cGLM{1}.prmFits;
    mOpt.useDif = 1;
    mOpt.useGroups = 1;
    mOpt.contOnly = 0;
    mOpt.freeP = [1,1,1,0,1,1]>0;
    mOpt.crossVal = 1;
    mOpt.crossValFolds = 2;
    mOpt.groupIDs = testSite;

    [~, selfFit] = uniBlks.getModelFitsToInactivationData(mOpt);
    logLikContTest(i,1) = mean(selfFit{1}.logLik);
    mOpt.freeP = [0,0,0,0,0,0]>0;
    [~, contFit] = uniBlks.getModelFitsToInactivationData(mOpt);
    logLikContTest(i,2) = mean(contFit{1}.logLik);
end

[~, pVal] = ttest(logLikContTest(:,1),logLikContTest(:,2));
pVal = round(pVal, 2, 'significant');
%%
uniBlks = spatialAnalysis('all', 'uniscan', 1, 1);

% Get the control fits
mOpt = struct;
mOpt.contOnly = 1;
mOpt.nRepeats = 10;
mOpt.useDif = 0;
cGLMs = uniBlks.getModelFitsToInactivationData(mOpt);
fprintf('Done control shuffles... \n');

% Get the fits for the laser trials
mOpt.contParams = mean(cell2mat(cellfun(@(x) x.prmFits, cGLMs, 'uni', 0)));
mOpt.useDif = 1;
mOpt.useGroups = 1;
mOpt.contOnly = 0;
mOpt.freeP = [1,1,1,0,1,1]>0;
mOpt.groupIDs = testSite;
[~, tstGLMs] = uniBlks.getModelFitsToInactivationData(mOpt);
fprintf('Done inactivation shuffles... \n');

% Get the "grids" of behavioural information for each shuffle
glmsCont_Test = [{cGLMs}; tstGLMs];
gridsTest = cellfun(@(x) prc.getGridsFromBlock(x.blockData), glmsCont_Test{2}, 'uni', 0);
fracRightTurns = squeeze(mean(cell2mat(cellfun(@(x) permute(x.fracRightTurns,[3,1,2]), gridsTest, 'uni', 0)),1));

%account for trial with 100% right turns
satIdx = fracRightTurns==1;
numTrialsAtSaturation = mean(cell2mat(cellfun(@(x) x.numTrials(satIdx), gridsTest, 'uni', 0)));
fracRightTurns(satIdx) = numTrialsAtSaturation/(numTrialsAtSaturation+1);
fracRightTurns = log10(fracRightTurns./(1-fracRightTurns));
fprintf('Done behaviour grid extraction... \n');

figure;
box off
hold on
for i = 1:2
    glm2Plot = glmsCont_Test{i}{1};
    glm2Plot.prmFits = mean(cell2mat(cellfun(@(x) x.prmFits,glmsCont_Test{i}, 'uni', 0)));
    if i == 1
        glm2Plot.prmFits = mOpt.contParams;
    else
        glm2Plot.prmFits = mean(cell2mat(cellfun(@(x) x.prmFits,glmsCont_Test{i}, 'uni', 0)));
    end

    pHatCalculated = glm2Plot.calculatepHat(glm2Plot.prmFits,'eval');

    [grids.visValues, grids.audValues] = meshgrid(unique(glm2Plot.evalPoints(:,1)),unique(glm2Plot.evalPoints(:,2)));
    [~, gridIdx] = ismember(glm2Plot.evalPoints, [grids.visValues(:), grids.audValues(:)], 'rows');
    plotData = grids.visValues;
    plotData(gridIdx) = pHatCalculated(:,2);

    plotData = log10(plotData./(1-plotData));
    contrastPower = mOpt.contParams(4);

    if i == 1
        plotOpt.lineStyle = '--';
        plotOpt.lineWidth = 0.75;
    else
        plotOpt.lineStyle = '-';
        plotOpt.lineWidth = 2;
    end
    plotOpt.Marker = 'none';

    visValues = (abs(grids.visValues(1,:))).^contrastPower.*sign(grids.visValues(1,:));
    lineColors = plt.selectRedBlueColors(grids.audValues(:,1));
    plt.rowsOfGrid(visValues, plotData, lineColors, plotOpt);

    if i == 2
        plotOpt.lineStyle = 'none';
        plotOpt.Marker = '.';
        visValues = gridsTest{1}.visValues(1,:);
        maxContrast = max(abs(visValues(1,:)));
        visValues = abs(visValues./maxContrast).^contrastPower.*sign(visValues);
        plt.rowsOfGrid(visValues, fracRightTurns, lineColors, plotOpt);

        xTickLoc = (-1):(1/8):1;
        xTickLoc = sign(xTickLoc).*abs(xTickLoc).^contrastPower;

        set(gca, 'xTick', xTickLoc, 'xTickLabel', {'-80' '' '' '' '-40'  '' '' '' '0' '' '' '' '40' '' '' '' '80'});
    end
end
ylim([-2.5 2.5])
xlim([-1 1])
midPoint = 0;
xL = xlim; hold on; plot(xL,[midPoint midPoint], '--k', 'linewidth', 1.5);
yL = ylim; hold on; plot([0 0], yL, '--k', 'linewidth', 1.5);

ylabel('p(Rightward)/p(Leftward)')
xlabel('Visual contrast')
set(gca, 'YTickLabel', [0.01, 0.1, 1, 10, 100])
axis square;
title('Frontal inactivation');
text(max(xlim)*0.5, min(ylim)+0.5, ['P ~' num2str(pVal)]);
end