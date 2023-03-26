function comBlks = combineBlocks(blks)
%% Function to combine an array of block files into a universal format that can then be easily filtered and used for plotting

%INPUTS(default values)
%blks(required)---The struct array of block files to be combined

%OUTPUTS
%comBlks----------A struct array of blk files, now in the new format. Includes...
%   .tot-------------Struct with details about the total numbers in the combined block file
%       .subjects-----------------Total subjects
%       .experiments--------------Total experiments
%       .trials-------------------Total trials
%       .penetrations-------------Total penetrations (ephys only)
%       .clusters-----------------Total clusters (ephys only)
%   .exp-------------Struct where each field has a row for each experiment, with the following information about that experiment
%       .subjectRef---------------A numerical index indicating which subject each experiment came from
%       .subject------------------The name of the subject 
%       .expDate------------------Date
%       .expNum-------------------Expriment number (folder)
%       .rigName------------------Name of the rig
%       .expType------------------Type of the experiment (e.g. training, ephys, inactivation)
%       .expDef-------------------The name of the subject 
%       .conditionParametersAV----Cell with [audDiff visDiff] combinations
%       .conditionLabels----------Cell with label indices for each AV condition
%       .performanceAVM-----------Performance on [auditory visual coherent multisensory] trials
%       .numOfTrials--------------Number of trials
%   .tri-------------Struct where each field has a row for each trial, with the following information about that trial
%
%NEED TO FINISH THIS DESCRIPTION AND ALSO ADD TO HEADING OF SPATIALANALYSIS FUNCTION

%% Get initial information about numbers, and determine if ephys data exists (and how much there is)

%Get the unique subjects, a numerical index indicating which experiments come from each subject
[uniqueSubjects, ~, expBySubject] = unique({blks.subject}');

%Change storage method depending on number of blks to save space
if length(blks) > 255; storeAs1 = 'uint16'; else, storeAs1 = 'uint8'; end

%Get number of trials, and indices indicating which trial comes from which subject/expriment
numOfTrials = arrayfun(@(x) size(x.timings.trialStartEnd(:,1),1), blks);
trialBySubject = cell2mat(arrayfun(@(x,y) x*ones(y,1, 'uint8'), expBySubject, numOfTrials, 'uni', 0));
trialByExp = cell2mat(arrayfun(@(x,y) x*ones(y,1, storeAs1), cumsum(expBySubject*0+1), numOfTrials, 'uni', 0));

%Determine if ephys data is available, and if so, how many penetration sites are contained across all blocks
ephysAvailable = arrayfun(@(x) isfield(x, 'ephys'), blks);
numOfSites = arrayfun(@(x) length(x.ephys), blks(ephysAvailable));
if any(numOfSites); ephysExists = 1; else, ephysExists = 0; end

%% Process the ephys information if it is available
if ephysExists
    %Change storage method depending on number of penetration sites to save space
    if sum(numOfSites) > 255; storeAs2 = 'uint16'; else, storeAs2 = 'uint8'; end
    
    %Count the number of clusters per penetration and number of spikes per cluster
    ephys = vertcat(blks(numOfSites>0).ephys);
    clustersPerPen = arrayfun(@(x) length(x.cluster.amplitudes),ephys);
    spksPerCluster = cell2mat(arrayfun(@(x) cellfun(@length, x.cluster.spkTimes),ephys, 'uni', 0));
    
    %Concatenate the penetration data, record number of clusters, and add reference indices for the subject and experiment
    ephys = prc.catStructs(ephys);
    ephys.penetration.subjectRef = cell2mat(arrayfun(@(x,y) x*ones(y,1), expBySubject, numOfSites, 'uni', 0));
    ephys.penetration.expRef = cell2mat(arrayfun(@(x,y) x*ones(y,1), cumsum(numOfSites*0+1), numOfSites, 'uni', 0));
    ephys.penetration.numOfClusters = clustersPerPen;
    
    %AFter concatenation, get a reference index for each penetration and cluster: experiment, subjects, etc.
    penByExp = ephys.penetration.expRef;
    ephys.cluster.numOfSpikes = spksPerCluster;
    ephys.cluster.subjectRef = cell2mat(arrayfun(@(x,y) x*ones(y,1, 'uint8'), ephys.penetration.subjectRef, clustersPerPen, 'uni', 0));
    ephys.cluster.expRef = cell2mat(arrayfun(@(x,y) x*ones(y,1, storeAs1), penByExp, clustersPerPen, 'uni', 0));
    ephys.cluster.penetrationRef = cell2mat(arrayfun(@(x,y) x*ones(y,1, storeAs2), cumsum(penByExp*0+1), clustersPerPen, 'uni', 0));   
    
    %Get the experimental details for each penetration. Location of the probe, hemisphere, etc.
    expDets = kil.getExpDets(uniqueSubjects(ephys.penetration.subjectRef), {blks(penByExp).expDate}', {blks(penByExp).expNum}', ephys.penetration.folder);
    expDets = prc.catStructs(expDets);
    
    %Copy all expDets fields to "ephys.penetration" and then remove redundant basic information
    fields2copy = fields(expDets); 
    for i = fields2copy'; ephys.penetration.(i{1}) = expDets.(i{1}); end
    ephys.penetration = prc.chkThenRemoveFields(ephys.penetration, {'subject'; 'expDate'; 'expNum'; 'estDepth'});
end

%% Process information for the "tot" field
comBlks.tot.subjects = length(uniqueSubjects);
comBlks.tot.experiments = length(blks);
comBlks.tot.trials = sum(numOfTrials);
if ephysExists
    comBlks.tot.penetrations = sum(numOfSites);
    comBlks.tot.clusters = sum(clustersPerPen);
end

%% Process information for the "exp" field
%Add hardcoded set of fields from the blks to the exp field, including performanceAVM only on non-passive trials
comBlks.exp.subjectRef = expBySubject;
perExpFields = {'subject', 'expDate', 'expNum', 'rigName', 'expType', 'expDef', 'conditionParametersAV', 'conditionLabels'};
if ~any(contains({blks.expDef}', 'Passive')); perExpFields = [perExpFields, 'performanceAVM', 'inactivationSites', 'wheelTicksToDecision']; end
for i = perExpFields; comBlks.exp.(i{1}) = {blks.(i{1})}'; end
comBlks.exp.numOfTrials = numOfTrials;

%% Process information for the "tri" field
%Remove fields from the orignal "blks" array that aren't needed
blks = prc.chkThenRemoveFields(blks, [perExpFields, 'params', 'ephys', 'grids']);

%Add basic reference indices information for the subject and experiment of each trial
comBlks.tri.subjectRef = trialBySubject;
comBlks.tri.expRef = trialByExp;

%Check if timeline infomration is available. If it is, create a "timeline" field and add the information to the block
if isfield(blks, 'timeline') && any(arrayfun(@(x) ~isempty(x.timeline), blks))
    %Find experiments with available timeline data
    timelineAvailable = arrayfun(@(x) isfield(x, 'timeline') & ~isempty(x.timeline), blks);
    
    %Create "nanTimeline" which contains a single nan entry for each timeline field of the corresponding data type (e.g. cell, vector, etc.)
    nanTimeline = blks(find(timelineAvailable,1)).timeline;
    timelineFields = fields(nanTimeline);
    for i = timelineFields'
        if iscell(nanTimeline.(i{1})); nanTimeline.(i{1}) = {NaN};
        else, nanTimeline.(i{1}) = nanTimeline.(i{1})(1,:)*NaN;
        end
    end
    %Go through experiments with no timeline and fill them with nan values of the appropriate data type and length (number of trials)
    for i = find(~timelineAvailable)'
        for j = timelineFields'; blks(i).timeline.(j{1}) = repmat(nanTimeline.(j{1}), numOfTrials(i), 1); end
    end
    %Remove unwanted fields from timeline
    for i = 1:length(blks); blks(i).timeline = prc.chkThenRemoveFields(blks(i).timeline, {'alignment';'frameTimes'}); end
else, blks = prc.chkThenRemoveFields(blks, {'timeline'});
end

%Once timeline has been processed to account for missing timelines, we can now concatenate the original blk array
catBlks = prc.catStructs(blks);

%Add "trialFields" from the newly concatenated "blks" to "comBlks"
trialFields = {'trialType','trialClass', 'timings', 'timeline', 'stim', 'inactivation', 'outcome', 'raw'};
for i = trialFields
    if isfield(blks, i{1}); comBlks.tri.(i{1}) = catBlks.(i{1}); end
end
blks = prc.chkThenRemoveFields(blks, trialFields);

%% Add the "pen" and "clu" fields from the ephys data if it exists
if ephysExists
    comBlks.pen = ephys.penetration;
    comBlks.clu = ephys.cluster; 
end

%% Blks should now be empty, so through a warning if it isn't
if ~isempty(fields(blks)); warning('Unexpected fields in blocks!'); end
end




