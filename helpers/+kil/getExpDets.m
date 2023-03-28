function expDets = getExpDets(subject, expDate, expNum, folder)
%% Function to get the experimental details for a penetration based on the subject, date, expNum, and folder

%INPUTS(default values)
%subject(required)----------Name of subject
%expDate(required)----------Date of recording
%expNum(required)-----------Experiment number for recording
%folder(required)-----------Folder that ephys data was stored in (site1, site2, etc.)

%OUTPUTS
%expDets--------------------Structure with details of the penetration
%                .subject--------Name of subject
%                .expDate--------Experiment date
%                .expNum---------Experiment number
%                .estDepth-------Estimated depth when inserting probe
%                .histIdx--------Histology index (the index the penetration was given in histological reconstruction)
%                .folder---------Folder the data was stored in
%                .hemisphere-----Hemisphere of penetration ('R' or 'L')
%                .probePainted---Logical indicating if the probe was painted
%                .calcLine-------The calculated (from histology) direction vector of the probe in allen space
%                .calcTip--------The calculated (from histology) tip of the probe in allen space
%                .scalingFactor--The scaling factor to adjust "size" of probe for brain shrinkage
%                .ephysRecordIdx-The row in the excel sheet that the expDets were collected from

%%
%Make sure all inputs were cells (because of cellfun used below)
if ischar(subject); subject = {subject}; end
if ischar(expDate); expDate = {expDate}; end
if ischar(expNum); expNum = {expNum}; end
if ischar(folder); folder = {folder}; end
if contains(subject, expDate); subject{1} = subject{1}(1:5); end %Deals with case where date is appended to name for individual processing.

%Load the ephys record, and get cell arrays of the contained subjects, dates, exp numbers, and folders
ephysRecord = load(prc.pathFinder('ephysrecord')); ephysRecord = ephysRecord.ephysRecord;

allSubjects = {ephysRecord.subject};
allDates = {ephysRecord.expDate};
allExpNums = {ephysRecord.expNum};
allFolders = {ephysRecord.folder};

%Put the records from the excel sheet, and the requested records, into the same form for comparison. "all" accounts for cases where the same
%penetration was used for all experiment numbers. This is usually the case, but not always.
uniqueRecords = cellfun(@(w,x,y,z) lower([w,x,y,z]), allSubjects(:), allDates(:), allExpNums(:), allFolders(:), 'uni', 0);
requestedRecords = cellfun(@(w,x,y,z) lower([w,x,y,z]), subject, expDate, expNum, folder, 'uni', 0);
requestedRecordsAll = cellfun(@(w,x,y,z) lower([w,x,y,z]), subject, expDate, repmat({'all'}, length(folder),1), folder, 'uni', 0);

%Find the selected index in the sheet (checking both "all" and expNum specific records") and add records to the output
[~, selectedIdx] = ismember(requestedRecords, uniqueRecords);
[~, selectedIdxAll] = ismember(requestedRecordsAll, uniqueRecords);
selectedIdx = max([selectedIdx selectedIdxAll], [], 2);
expDets = ephysRecord(selectedIdx(selectedIdx>0));
for i = 1:length(expDets); expDets(i).ephysRecordIdx = selectedIdx(i); end
end
