function [contGLMs, deltaGLMs] = getModelFitsToInactivationData(obj, mOpt)
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

if ~exist('mOpt', 'var'); mOpt = struct; end
if ~isfield(mOpt, 'contOnly'); contOnly = 0; else; contOnly = mOpt.contOnly; end
if ~isfield(mOpt, 'useGroups'); useGroups = 0; else; useGroups = mOpt.useGroups; end
if ~isfield(mOpt, 'groupIDs'); groupIDs = ''; else; groupIDs = mOpt.groupIDs; end
if ~isfield(mOpt, 'crossVal'); crossVal = 0; else; crossVal = mOpt.crossVal; end
if ~isfield(mOpt, 'freeP'); freeP = 0; else; freeP = mOpt.freeP; end
if ~isfield(mOpt, 'nRepeats'); nRepeats = 0; else; nRepeats = mOpt.nRepeats; end
if ~isfield(mOpt, 'useDif'); useDif = 1; else; useDif = mOpt.useDif; end
if ~isfield(mOpt, 'crossValFolds'); crossValFolds = 5; else; crossValFolds = mOpt.crossValFolds; end

%Create "iBlk" (initialBlock) which removes some incorrect galvoPositions, repeated trials, and keeps only valid trials
iBlk = prc.filtBlock(obj.blks, obj.blks.tri.inactivation.galvoPosition(:,2)~=4.5);
iBlk = prc.filtBlock(iBlk, ~ismember(abs(iBlk.tri.inactivation.galvoPosition(:,1)),[0.5; 2; 3.5; 5]) | iBlk.tri.inactivation.laserType==0);
iBlk = prc.filtBlock(iBlk, iBlk.tri.trialType.repeatNum==1 & iBlk.tri.trialType.validTrial);
iBlk = prc.filtBlock(iBlk, ~isnan(iBlk.tri.outcome.responseCalc));

%Conditional to optionally group inactivaiton sites together (e.g. if you want to combine all V1 sites). We "reflect" these groups, so only one
%hemisphere needs to be defined. We find all trials with galvoPositions belonging to those groups in iBlk and replace with the mean group position
if useGroups
    galvoGrps = {};
    groups = lower(groupIDs);
    if contains(groups, 'v1'); galvoGrps = [galvoGrps; [1.8 -4; 3,-4; 3,-3]]; end
    if contains(groups, 's1'); galvoGrps = [galvoGrps; [3,1; 3,0; 4.2,0]]; end
    if contains(groups, 'a1'); galvoGrps = [galvoGrps; [4.2,-2; 4.2,-3; 4.2,-4]]; end
    if contains(groups, 'mos'); galvoGrps = [galvoGrps; [0.6 2; 1.8, 2; 0.6, 3]]; end
    if contains(groups, 'av'); galvoGrps = [galvoGrps; [1.8 -4; 3,-4; 4.2,-4; 1.8,-3; 3,-3; 4.2,-3; 1.8,-2; 3,-2; 4.2,-2]]; end
    
    galvoGrps = [galvoGrps; cellfun(@(x) [x(:,1)*-1, x(:,2)], galvoGrps, 'uni', 0)];
    grpNum = length(galvoGrps);
    grpIdx = cellfun(@(x,y) ismember(iBlk.tri.inactivation.galvoPosition, x, 'rows').*y, galvoGrps, num2cell(1:grpNum)', 'uni', 0);
    grpIdx = sum(cell2mat(grpIdx'),2);
    meanPositions = cellfun(@mean, galvoGrps, 'uni', 0);
    iBlk.tri.inactivation.galvoPosition(grpIdx>0,:) = cell2mat(meanPositions(grpIdx(grpIdx>0)));
    iBlk = prc.filtBlock(iBlk, iBlk.tri.inactivation.laserType==0 | grpIdx>0);
end


if useDif
    %If plotType contains 'dif' then we want to switch the responseCalc, vis and aud paramters such that all "left" trials are flipped (vis left trials in
    %the case of conflict trials). Now, inactivations on the right hemisphere are contralateral, and left hemisphere is ipsilateral
    idx2Flip = iBlk.tri.inactivation.galvoPosition(:,1)<0;
    iBlk.tri.outcome.responseCalc(idx2Flip) = (iBlk.tri.outcome.responseCalc(idx2Flip)*-1+3).*(iBlk.tri.outcome.responseCalc(idx2Flip)>0);
    iBlk.tri.inactivation.galvoPosition(idx2Flip,1) = -1*iBlk.tri.inactivation.galvoPosition(idx2Flip,1);
    iBlk.tri.stim.audDiff(idx2Flip) = -1*iBlk.tri.stim.audDiff(idx2Flip);
    iBlk.tri.stim.visDiff(idx2Flip) = -1*iBlk.tri.stim.visDiff(idx2Flip);
    iBlk.tri.stim.conditionLabel(idx2Flip) = -1*iBlk.tri.stim.conditionLabel(idx2Flip);
end

contBlk = prc.filtBlock(iBlk, iBlk.tri.inactivation.laserType == 0);
uniBlk = prc.filtBlock(iBlk, iBlk.tri.inactivation.laserType == 1);
[~, gridXY] = prc.makeGrid(uniBlk, uniBlk.tri.outcome.responseCalc, [], 'galvouni',2);

%Use some matrix tricks to create "uniformLaserFilters" which is a filter for each shuffle that equalizes the frequency of subjects
%contributing to each point in the grid of galvo positions.
trialIdx = (1:uniBlk.tot.trials)';

if ~contOnly
    nonUniformLaser = prc.makeGrid(uniBlk, [double(uniBlk.tri.subjectRef) trialIdx], [], 'galvouni',2);
    laserShuffles = cellfun(@(x) double(prc.makeFreqUniform(x(:,1),nRepeats,x(:,2))), nonUniformLaser, 'uni', 0);
    laserShuffles = num2cell(cell2mat(laserShuffles(:)),1)';
    uniformLaserFilters = cellfun(@(x) ismember(trialIdx, x), laserShuffles, 'uni', 0);
end

if ~isfield(mOpt, 'contParams')
    uniformControlFilters = num2cell(prc.makeFreqUniform(contBlk.tri.subjectRef,nRepeats),1)';
end

%Removing these excess fields makes "filtBlock" run significantly faster in the subsequent loop
uniBlk.tri = rmfield(uniBlk.tri, {'timings'});
contBlk.tri = rmfield(contBlk.tri, {'timings'});
uniBlk.tri.inactivation = rmfield(uniBlk.tri.inactivation, {'laserPower', 'galvoType', 'laserOnsetDelay', 'laserDuration'});
contBlk.tri.inactivation = rmfield(contBlk.tri.inactivation, {'laserPower', 'galvoType', 'laserOnsetDelay', 'laserDuration'});

%This is the same loop as above, to generate "inactiveGrid" but with some extra steps to deal with the shuffles
contGLMs = cell(nRepeats,1);
deltaGLMs = cell(size(gridXY{1}));
for i = 1:nRepeats
    %Filter both blks to make contributions from each subject equal at each galvoPosition and across control trials.
    if ~isfield(mOpt, 'contParams')
        subContBlk = prc.filtBlock(contBlk, uniformControlFilters{i});
        contGLM = fit.GLMmulti(subContBlk, 'simpLogSplitVSplitA');
        contGLM.fit;
        contGLMs{i} = contGLM;
        contParams = contGLM.prmFits;
    else
        contParams = mOpt.contParams;
    end
    if contOnly; continue; end
    subTestBlk = prc.filtBlock(uniBlk, uniformLaserFilters{i});
    
    for j = 1:length(gridXY{1}(:))
        galvoRef = [gridXY{1}(j) gridXY{2}(j)];
        includeIdx = ismember(subTestBlk.tri.inactivation.galvoPosition,galvoRef, 'rows');
        subTestGalvoBlk = prc.filtBlock(subTestBlk, includeIdx);
        
        deltaGLM = fit.GLMmulti(subTestGalvoBlk, 'simpLogSplitVSplitA');
        deltaGLM.prmInit = contParams;
        deltaGLM.blockData.freeP = freeP;
        if crossVal; deltaGLM.fitCV(crossValFolds); else;  deltaGLM.fit; end
        if nRepeats == 1; deltaGLMs{j} = deltaGLM;
        else, deltaGLMs{j}{i,1} = deltaGLM;
        end
    end
end
end