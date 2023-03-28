function inactCompResults = inactivationPairwiseLLRSiteVsShuffle(obj, nShuffles, pInclude)
%% Method for "spatialAnalysis" class. Plots effects of inactivation on behavior. Plots are shown as grids on an outline of cortex.

%INPUTS(default values)
%plotType(res)---------A string that contains three letter tags indicates the type of plot to generate. Can be combination of below options
%   'res'-------------------------quantify changes in the mouse response (i.e. fraciton of rightward choices)
%   'dif'-------------------------rather than separately analysing left and right trials, combine trials and use ipsilateral and contralateral
%   'grp'-------------------------combine inactivation sites into M2 and Vis
%   'sig'-------------------------test the significance of inactivation by shuffling inactivation sites and laser on/off
%nShuffles-------------The number of times to shuffle te data
%subsets---------------The data subsets to make plots for (see function "prc.getDefinedSubset")

%Set up defaults for the input values. "op2use" is "mean" as default for responses, but changes below depending on the type data being used
numOfMice = length(obj.blks);
if numOfMice > 1; error('Only coded to handle one mouse atm'); end
if ~exist('nShuffles', 'var'); nShuffles = 0; end
if ~exist('pInclude', 'var'); pInclude = 1:5; end
freeP = [1 1 1 1 1 1]>0;

%Create "iBlk" (initialBlock) which removes some incorrect galvoPositions, repeated trials, and keeps only valid trials
iBlk = prc.filtBlock(obj.blks, obj.blks.tri.inactivation.galvoPosition(:,2)~=4.5);
iBlk = prc.filtBlock(iBlk, ~ismember(abs(iBlk.tri.inactivation.galvoPosition(:,1)),[0.5; 2; 3.5; 5]) | iBlk.tri.inactivation.laserType==0);
iBlk = prc.filtBlock(iBlk, iBlk.tri.trialType.repeatNum==1 & iBlk.tri.trialType.validTrial);
iBlk = prc.filtBlock(iBlk, ~isnan(iBlk.tri.outcome.responseCalc));
iBlk.tri.inactivation.galvoPosition(iBlk.tri.inactivation.laserType==0,:) = repmat([0 0],sum(iBlk.tri.inactivation.laserType==0),1);

%Conditional to optionally group inactivaiton sites together (e.g. if you want to combine all V1 sites). We "reflect" these groups, so only one
%hemisphere needs to be defined. We find all trials with galvoPositions belonging to those groups in iBlk and replace with the mean group position
posNames = {'MOs'; 'V1'; 'A1'; 'S1';'None'};
prmLabels = {'Bias'; 'visScaleIpsi'; 'visScaleConta'; 'N'; 'audScaleIpsi'; 'audScaleContra'};
posNames = posNames(pInclude);
galvoGrps = {[0.6 2; 1.8, 2; 0.6, 3];[1.8 -4; 3,-4; 3,-3];[4.2,-2; 4.2,-3; 4.2,-4];[3,1; 3,0; 4.2,0];[0 0 ; 0 0]};
galvoGrps = galvoGrps(pInclude);

tstDat = [abs(iBlk.tri.inactivation.galvoPosition(:,1)) iBlk.tri.inactivation.galvoPosition(:,2)];
grpNum = length(galvoGrps);
grpIdx = cellfun(@(x,y) ismember(tstDat, x, 'rows').*y, galvoGrps, num2cell(1:grpNum)', 'uni', 0);
grpIdx = sum(cell2mat(grpIdx'),2);
meanPositions = cellfun(@mean, galvoGrps, 'uni', 0);
iBlk.tri.inactivation.grpIdx = grpIdx;
galvoSign = iBlk.tri.inactivation.galvoPosition(:,1)<0;
iBlk.tri.inactivation.galvoPosition(grpIdx>0,:) = cell2mat(meanPositions(grpIdx(grpIdx>0)));
iBlk.tri.inactivation.galvoPosition(galvoSign,1) = iBlk.tri.inactivation.galvoPosition(galvoSign,1)*-1;
iBlk = prc.filtBlock(iBlk, iBlk.tri.inactivation.laserType==0 | grpIdx>0);

nBlk = iBlk;

%If plotType contains 'dif' then we want to switch the responseCalc, vis and aud paramters such that all "left" trials are flipped (vis left trials in
%the case of conflict trials). 
idx2Flip = nBlk.tri.inactivation.galvoPosition(:,1)<0;
nBlk.tri.outcome.responseCalc(idx2Flip) = (nBlk.tri.outcome.responseCalc(idx2Flip)*-1+3).*(nBlk.tri.outcome.responseCalc(idx2Flip)>0);
nBlk.tri.inactivation.galvoPosition(idx2Flip,1) = -1*nBlk.tri.inactivation.galvoPosition(idx2Flip,1);
nBlk.tri.stim.audDiff(idx2Flip) = -1*nBlk.tri.stim.audDiff(idx2Flip);
nBlk.tri.stim.visDiff(idx2Flip) = -1*nBlk.tri.stim.visDiff(idx2Flip);
nBlk.tri.stim.conditionLabel(idx2Flip) = -1*nBlk.tri.stim.conditionLabel(idx2Flip);

%Define number of suffles, and the number of times to estimate the control (default is 10% or 500) since this will change with different
%subsamples of the subjetcs. Total loops is the sum of shuffle and control loops. We randomly assign galvoPositions to the contBlk from the
%galvoPositions in the testBlock (for shuffling purposes)
if ~nShuffles; nShuffles = 1500; end
nShuffles = round(nShuffles/10)*10;
normEstRepeats = round(nShuffles/10);

%Use some matrix tricks to create "uniformLaserFilters" which is a filter for each shuffle that equalizes the frequency of subjects
%contributing to each point in the grid of galvo positions.
trialIdx = (1:nBlk.tot.trials)';
nonUniformLaser = prc.makeGrid(nBlk, [double(nBlk.tri.subjectRef) trialIdx], [], 'galvouni',2);

laserShuffles = cellfun(@(x) double(prc.makeFreqUniform(x(:,1),normEstRepeats,x(:,2))), nonUniformLaser, 'uni', 0);
maxTrials = cellfun(@length, laserShuffles(:));
maxTrials = floor(min(maxTrials(maxTrials > 0)/nBlk.tot.subjects)/2)*2;

laserShuffles = cellfun(@(x) double(prc.makeFreqUniform(x(:,1),normEstRepeats,x(:,2),maxTrials)), nonUniformLaser, 'uni', 0);
laserShuffles = num2cell(cell2mat(laserShuffles(:)),1)';
uniformLaserFilters = repmat(cellfun(@(x) ismember(trialIdx, x), laserShuffles, 'uni', 0),11,1);

%Removing these excess fields makes "filtBlock" run significantly faster in the subsequent loop
nBlk.tri = rmfield(nBlk.tri, {'timings'});
nBlk.tri.inactivation = rmfield(nBlk.tri.inactivation, {'laserPower', 'galvoType', 'laserOnsetDelay', 'laserDuration'});

%This is the same loop as above, to generate "inactiveGrid" but with some extra steps to deal with the shuffles

totalLoops = normEstRepeats+nShuffles;
[grpIdx1, grpIdx2] = meshgrid(1:5, 1:5);
grpOrd = [grpIdx1(:) grpIdx2(:)];
grpOrd(diff(grpOrd,[],2)==0,:) = [];

tDat = nBlk.tri.inactivation.galvoPosition;
mPos = cell2mat(meanPositions);
grpFilter = cellfun(@(x) ismember(tDat, mPos([x(1) x(2)],:), 'rows'), num2cell(grpOrd,2), 'uni', 0);

[trainGrp1Params, trainGrp2Params, trainGrp1LogLik, trainGrp2LogLik,  ...
    testGrp1Params, testGrp2Params, testGrp1LogLik, testGrp2LogLik] = deal(cell(size(grpOrd,1),1));
for j = 1:size(grpOrd,1)
    for i = 1:totalLoops
        if grpOrd(j,1) > grpOrd(j,2); continue; end
        altIdx = find(grpOrd(:,1) == grpOrd(j,2) & grpOrd(:,2) == grpOrd(j,1));
        %Filter both blks to make contributions from each subject equal at each galvoPosition and across control trials.
        uniformBlk = prc.filtBlock(nBlk, uniformLaserFilters{i} & grpFilter{j});
        
        %Generate "randomBlk" by concatenating control and test blocks. If the repeat number is outside the "normEstRepeats" range then we shuffle
        %the laser activation and galvoPositions
        if i > normEstRepeats
            for k = 1:uniformBlk.tot.subjects
                sIdx = find(uniformBlk.tri.subjectRef == k);
                uniformBlk.tri.inactivation.grpIdx(sIdx,:) = uniformBlk.tri.inactivation.grpIdx(sIdx(randperm(length(sIdx))),:);
            end
        end
        grp1Blk = prc.filtBlock(uniformBlk, uniformBlk.tri.inactivation.grpIdx == grpOrd(j,1));
        grp2Blk = prc.filtBlock(uniformBlk, uniformBlk.tri.inactivation.grpIdx == grpOrd(j,2));
        
        randSplit = prc.makeFreqUniform(grp1Blk.tri.subjectRef,1,[],grp1Blk.tot.trials/grp1Blk.tot.subjects/2);
        trainGrp1Blk = prc.filtBlock(grp1Blk, randSplit);
        testGrp1Blk = prc.filtBlock(grp1Blk, ~randSplit);
        
        randSplit = prc.makeFreqUniform(grp2Blk.tri.subjectRef,1,[],grp2Blk.tot.trials/grp2Blk.tot.subjects/2);
        trainGrp2Blk = prc.filtBlock(grp2Blk, randSplit);
        testGrp2Blk = prc.filtBlock(grp2Blk, ~randSplit);
       
        
        trainGrp1GLM = fit.GLMmulti(trainGrp1Blk, 'simpLogSplitVSplitA');
        trainGrp1GLM.blockData.freeP = freeP;
        trainGrp1GLM.fit
        trainGrp1Params{j}(i,:) = trainGrp1GLM.prmFits;
        trainGrp1LogLik{j}(i,:) = mean(trainGrp1GLM.logLik);
        
        trainGrp2Params{altIdx}(i,:) = trainGrp1GLM.prmFits;
        trainGrp2LogLik{altIdx}(i,:) = mean(trainGrp1GLM.logLik);
        
        trainGrp2GLM = fit.GLMmulti(trainGrp2Blk, 'simpLogSplitVSplitA');
        trainGrp2GLM.blockData.freeP = freeP;
        trainGrp2GLM.fit
        trainGrp2Params{j}(i,:) = trainGrp2GLM.prmFits;
        trainGrp2LogLik{j}(i,:) = mean(trainGrp2GLM.logLik);
        
        trainGrp1Params{altIdx}(i,:) = trainGrp2GLM.prmFits;
        trainGrp1LogLik{altIdx}(i,:) = mean(trainGrp2GLM.logLik);
             
        testGrp1GLM = fit.GLMmulti(testGrp1Blk, 'simpLogSplitVSplitA');
        testGrp1GLM.prmInit = trainGrp1Params{j}(i,:);
        testGrp1GLM.blockData.freeP = (freeP*0)>0;
        testGrp1GLM.fit;
        testGrp1Params{j}(i,:) = mean(testGrp1GLM.prmFits,1);
        testGrp1LogLik{j}(i,:) = mean(testGrp1GLM.logLik);
        
        testGrp2GLM = fit.GLMmulti(testGrp1Blk, 'simpLogSplitVSplitA');
        testGrp2GLM.prmInit = trainGrp2Params{j}(i,:);
        testGrp2GLM.blockData.freeP = (freeP*0)>0;
        testGrp2GLM.fit;
        testGrp2Params{j}(i,:) = mean(testGrp2GLM.prmFits,1);
        testGrp2LogLik{j}(i,:) = mean(testGrp2GLM.logLik);
         
        
        testGrp1GLM = fit.GLMmulti(testGrp2Blk, 'simpLogSplitVSplitA');
        testGrp1GLM.prmInit = trainGrp2Params{j}(i,:);
        testGrp1GLM.blockData.freeP = (freeP*0)>0;
        testGrp1GLM.fit;
        testGrp1Params{altIdx}(i,:) = mean(testGrp1GLM.prmFits,1);
        testGrp1LogLik{altIdx}(i,:) = mean(testGrp1GLM.logLik);
        
        testGrp2GLM = fit.GLMmulti(testGrp2Blk, 'simpLogSplitVSplitA');
        testGrp2GLM.prmInit = trainGrp1Params{j}(i,:);
        testGrp2GLM.blockData.freeP = (freeP*0)>0;
        testGrp2GLM.fit;
        testGrp2Params{altIdx}(i,:) = mean(testGrp2GLM.prmFits,1);
        testGrp2LogLik{altIdx}(i,:) = mean(testGrp2GLM.logLik);
        
        disp([i j]);
    end
end
%%
inactCompResults.posNames = posNames;
inactCompResults.trainTestGroups = grpOrd;
inactCompResults.prmLabels = prmLabels;
inactCompResults.meanPositions = meanPositions;
inactCompResults.normEstRepeats = normEstRepeats;
inactCompResults.trainGrp1Params = trainGrp1Params;
inactCompResults.trainGrp2Params = trainGrp2Params;
inactCompResults.testGrp1Params = testGrp1Params;
inactCompResults.testGrp2Params = testGrp2Params;
inactCompResults.trainGrp1LogLik = trainGrp1LogLik;
inactCompResults.trainGrp2LogLik = trainGrp2LogLik;
inactCompResults.testGrp1LogLik = testGrp1LogLik;
inactCompResults.testGrp2LogLik = testGrp2LogLik;
%%
savePath = [prc.pathFinder('processeddirectory') 'XSupData\'];
savePath = [savePath 'figS4inactCompResults.mat'];
save(savePath, '-struct', 'inactCompResults')
end