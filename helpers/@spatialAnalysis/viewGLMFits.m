function viewGLMFits(obj, modelString, cvFolds, plotType, multiFit, contrastPower)
if ~exist('modelString', 'var') || isempty(modelString); modelString = 'simpLogSplitVSplitA'; end
if ~exist('cvFolds', 'var') || isempty(cvFolds); cvFolds = 0; end
if ~exist('plotType', 'var') || isempty(plotType); plotType = 'normal'; end
if ~exist('multiFit', 'var') || isempty(multiFit); multiFit = 1; end
if ~strcmpi(plotType, 'none'); noPlt = 0; else; noPlt = 1; end
if ~strcmpi(plotType, 'only'); onlyPlt = 0; else; onlyPlt = 1; end
if ~contains(plotType, 'control'); contPlot = 0; else; contPlot = 1; end

if ~onlyPlt; obj.glmFit = cell(length(obj.blks),1); end
recalcContrastPower = 0;
for i  = 1:length(obj.blks)   
    if ~onlyPlt
        normBlk = spatialAnalysis.getBlockType(obj.blks(i),'norm');
        if ~any(normBlk.tri.stim.audInitialAzimuth==0)
            normBlk.tri.stim.audInitialAzimuth(isinf(normBlk.tri.stim.audInitialAzimuth)) = 0;
            normBlk.tri.stim.audDiff(isinf(normBlk.tri.stim.audDiff)) = 0;
        end
        normBlk = prc.filtBlock(normBlk,~isinf(normBlk.tri.stim.audInitialAzimuth));
        disp(normBlk.tot.trials);
        uniFilters = num2cell(prc.makeFreqUniform(normBlk.tri.subjectRef,multiFit),1)';
        allBlockData = cellfun(@(x) prc.filtBlock(normBlk, x), uniFilters, 'uni', 0);
        obj.glmFit{i} = cellfun(@(x) fit.GLMmulti(x, modelString), allBlockData, 'uni', 0);
        allBlockData = prc.catStructs(cell2mat(allBlockData));
    else
        normBlk = obj.blks(i);
    end
    if ~onlyPlt
        if ~cvFolds
            cellfun(@(x) x.fit, obj.glmFit{i});
        end
        if cvFolds
            cellfun(@(x) x.fitCV(cvFolds), obj.glmFit{i});
        end
    end
    if noPlt; return; end
    
    params2use = mean(cell2mat(cellfun(@(x) x.prmFits, obj.glmFit{i}, 'uni', 0)),1);
    obj.glmFit{i} = obj.glmFit{i}{1};
    pHatCalculated = obj.glmFit{i}.calculatepHat(params2use,'eval');
    [grids.visValues, grids.audValues] = meshgrid(unique(obj.glmFit{i}.evalPoints(:,1)),unique(obj.glmFit{i}.evalPoints(:,2)));
    [~, gridIdx] = ismember(obj.glmFit{i}.evalPoints, [grids.visValues(:), grids.audValues(:)], 'rows');
    plotData = grids.visValues;
    plotData(gridIdx) = pHatCalculated(:,2);
    plotOpt.lineStyle = '-';
    if contPlot
        plotOpt.lineStyle = '--';
        plotOpt.lineWidth = 0.75;
    end
    plotOpt.Marker = 'none';
    
    if contains(plotType, 'log')
        if recalcContrastPower == 1 || ~exist('contrastPower', 'var')
            contrastPower  = params2use(strcmp(obj.glmFit{i}.prmLabels, 'N'));
            recalcContrastPower = 1;
        end
        if isempty(contrastPower)
            tempFit = fit.GLMmulti(normBlk, 'simpLogSplitVSplitA');
            tempFit.fit;
            tempParams = mean(tempFit.prmFits,1);
            contrastPower  = tempParams(strcmp(tempFit.prmLabels, 'N'));
        end
        plotData = log10(plotData./(1-plotData));
    else
        contrastPower = 1;
    end
    visValues = (abs(grids.visValues(1,:))).^contrastPower.*sign(grids.visValues(1,:));
    lineColors = plt.selectRedBlueColors(grids.audValues(:,1));
    plt.rowsOfGrid(visValues, plotData, lineColors, plotOpt);
    
    plotOpt.lineStyle = 'none';
    plotOpt.Marker = '.';
    
    visDiff = allBlockData.tri.stim.visDiff;
    audDiff = allBlockData.tri.stim.audDiff;
    responseCalc = allBlockData.tri.outcome.responseCalc;
    [visGrid, audGrid] = meshgrid(unique(visDiff),unique(audDiff));
    maxContrast = max(abs(visGrid(1,:)));
    fracRightTurns = arrayfun(@(x,y) mean(responseCalc(ismember([visDiff,audDiff],[x,y],'rows'))==2), visGrid, audGrid);
    
    visValues = abs(visGrid(1,:)).^contrastPower.*sign(visGrid(1,:))./(maxContrast.^contrastPower);
    if contains(plotType, 'log')
        fracRightTurns = log10(fracRightTurns./(1-fracRightTurns));
    end
    if ~contPlot
        plt.rowsOfGrid(visValues, fracRightTurns, lineColors, plotOpt);
    end

    xlim([-1 1])
    midPoint = 0.5;
    xTickLoc = (-1):(1/8):1;
    if contains(plotType, 'log')
        ylim([-2.6 2.6])
        midPoint = 0;
        xTickLoc = sign(xTickLoc).*abs(xTickLoc).^contrastPower;
    end
    
    box off;
    xTickLabel = num2cell(round(((-maxContrast):(maxContrast/8):maxContrast)*100));
    xTickLabel(2:2:end) = deal({[]});
    set(gca, 'xTick', xTickLoc, 'xTickLabel', xTickLabel);
    %%
    title([obj.blks(i).exp.subject{1} '    n = ' num2str(normBlk.tot.trials)]);
    xL = xlim; hold on; plot(xL,[midPoint midPoint], '--k', 'linewidth', 1.5);
    yL = ylim; hold on; plot([0 0], yL, '--k', 'linewidth', 1.5);  
end
obj.hand.figure = [];
end