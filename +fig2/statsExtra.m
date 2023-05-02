function statsExtra
loadDir = [prc.pathFinder('processeddirectory') 'XSupData\'];
load([loadDir 'figS4bInactResultsForChoice_IndiMice'], 'inactResultsForChoice')

subRegions = {[0.6 2; 1.8, 2; 0.6, 3];[1.8 -4; 3,-4; 3,-3];[4.2,-2; 4.2,-3; 4.2,-4];...
    [-0.6 2; -1.8, 2; -0.6, 3];[-1.8 -4; -3,-4; -3,-3];[-4.2,-2; -4.2,-3; -4.2,-4]};
subRegions_Names = {'ContraMOs'; 'ContraVis'; 'ContraLateralSensory'; ...
    'IpsiMOs'; 'IpsiVis'; 'IpsiLateralSensory';};
subsets = inactResultsForChoice.subsets;
compareEffect_Reg_Subset = cell(length(subRegions),length(subsets));

gridX = inactResultsForChoice.gridXY{1}{1};
gridY = inactResultsForChoice.gridXY{1}{2};
for j = 1:5
    for i = 1:length(subsets)
        for k = 1:length(subRegions)
            contData = inactResultsForChoice.meanContEffects{i,j};
            dat2Take = gridXY*0;
            for q = 1:length(subRegions{k})
                dat2Take(gridX==subRegions{k}(q,1) & gridY==subRegions{k}(q,2)) = 1;
            end
            compareEffect_Reg_Subset{k,i}(j,1) = mean(contData(dat2Take>0));
        end
    end
end

[~, pVal] = ttest(compareEffect_Reg_Subset{2,1}, compareEffect_Reg_Subset{5,1});
pVal = round(pVal, 2, 'significant');
fprintf('Vis inactivation has weaker effects in coherent than visual trials: p < %f \n', pVal)

[~, pVal] = ttest(compareEffect_Reg_Subset{1,1}, compareEffect_Reg_Subset{1,2});
pVal = round(pVal, 2, 'significant');
fprintf('MOs inactivation was similar on both visual and auditory trials: p ~ %f \n', pVal)

[~, pVal] = ttest(compareEffect_Reg_Subset{3,1}, compareEffect_Reg_Subset{3,2});
pVal = round(pVal, 2, 'significant');
fprintf('Lateral sensory inactivation has a stronger effect on visual trials than auditory: p < %f \n', pVal)

%%
loadDir = [prc.pathFinder('processeddirectory') 'XSupData\'];
load([loadDir 'figS4bInactResultsForChoice_IndiMice'], 'inactResultsForChoice')

subRegions = {[0.6 2; 1.8, 2; 0.6, 3];[1.8 -4; 3,-4; 3,-3];[4.2,-2; 4.2,-3; 4.2,-4];...
    [-0.6 2; -1.8, 2; -0.6, 3];[-1.8 -4; -3,-4; -3,-3];[-4.2,-2; -4.2,-3; -4.2,-4]};
subRegions_Names = {'ContraMOs'; 'ContraVis'; 'ContraLateralSensory'; ...
    'IpsiMOs'; 'IpsiVis'; 'IpsiLateralSensory';};
subsets = inactResultsForChoice.subsets;
compareEffect_Reg_Subset = cell(length(subRegions),length(subsets));

gridX = inactResultsForChoice.gridXY{1}{1};
gridY = inactResultsForChoice.gridXY{1}{2};
for j = 1:5
    for i = 1:length(subsets)
        for k = 1:length(subRegions)
            contData = inactResultsForChoice.meanContEffects{i,j};
            dat2Take = gridXY*0;
            for q = 1:length(subRegions{k})
                dat2Take(gridX==subRegions{k}(q,1) & gridY==subRegions{k}(q,2)) = 1;
            end
            compareEffect_Reg_Subset{k,i}(j,1) = mean(contData(dat2Take>0));
        end
    end
end

[~, pVal] = ttest(compareEffect_Reg_Subset{2,1}, compareEffect_Reg_Subset{5,1});
pVal = round(pVal, 2, 'significant');
fprintf('Vis inactivation has weaker effects in coherent than visual trials: p < %f \n', pVal)

[~, pVal] = ttest(compareEffect_Reg_Subset{1,1}, compareEffect_Reg_Subset{1,2});
pVal = round(pVal, 2, 'significant');
fprintf('MOs inactivation was similar on both visual and auditory trials: p ~ %f \n', pVal)

[~, pVal] = ttest(compareEffect_Reg_Subset{3,1}, compareEffect_Reg_Subset{3,2});
pVal = round(pVal, 2, 'significant');
fprintf('Lateral sensory inactivation has a stronger effect on visual trials than auditory: p < %f \n', pVal)

%%
uniMice = {'PC027'; 'PC029'; 'DJ008'; 'DJ006'; 'DJ007'};
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

    mOpt.groupIDs = 'A1';
    [~, deltaFit] = uniBlks.getModelFitsToInactivationData(mOpt);
    VcChange(i,1) = mean(deltaFit{1}.prmFits(:,3));
    AcChange(i,1) = mean(deltaFit{1}.prmFits(:,6));
end

[~, pVal] = ttest(VcChange,AcChange);
pVal = round(pVal, 2, 'significant');
fprintf('Lateral sensory inactivation has a stronger effect on Vc param than Ac param: p < %f \n', pVal)

%%
uniMice = {'PC027'; 'PC029'; 'DJ008'; 'DJ006'; 'DJ007'};
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

    mOpt.groupIDs = 'MOs';
    [~, deltaFit] = uniBlks.getModelFitsToInactivationData(mOpt);
    VcChange(i,1) = mean(deltaFit{1}.prmFits(:,3));
    AcChange(i,1) = mean(deltaFit{1}.prmFits(:,6));
end

[~, pVal] = ttest(VcChange,AcChange);
pVal = round(pVal, 2, 'significant');
fprintf('MOs inactivation has a similar effects Vc and Ac params: p ~ %f \n', pVal)

end