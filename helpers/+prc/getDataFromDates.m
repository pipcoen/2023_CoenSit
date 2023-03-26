function data = getDataFromDates(subject, requestedDates, expDef, extraData)
%% Function to load proessed files from dates. Works with files from the convertExpFiles funciton.

%INPUTS(default values)
%subject(required)----------The subject for which the dates are requested
%requestedDates('last')-----A string representing the dates requested. Can be...
%                'yyyy-mm-dd'--------------A specific date
%                'all'---------------------All data
%                'lastx'-------------------The last x days of data (especially useful during training)
%                'firstx'------------------The first x days of date
%                'yest'--------------------The x-1 day, where x is the most recent day
%                'yyyy-mm-dd:yyyy-mm-dd'---Dates in this range (including the boundaries)
%expDef('multiSpaceWorld')--Specify the expDef to be loaded (otherwise blk files will no concatenate properly)
%extraData('eph')-----------The extra data requested in addition to the behavior. Can be...
%                'all'--------------------Both the ephys and raw data
%                'eph'--------------------Ephys data (spikes, etc. for each penetration)
%                'raw'--------------------Raw data (meaning the complete wheel trace etc. from the block files)

%OUTPUTS
%data-----------------------A struct array of blk files, with additional raw, timeline, and ephys data if requested

%% Check inputs are cells, assign defaults, load the expList (as defined by prc.scanForNewFiles)
if ~exist('subject', 'var'); error('Must specify subject'); end
if ~exist('requestedDates', 'var') || isempty(requestedDates); requestedDates = {'last'}; end
if ~exist('expDef', 'var'); expDef = 'multiSpaceWorld'; end
if ~exist('extraData', 'var'); extraData = 'eph'; end
if ~iscell(requestedDates); requestedDates = {requestedDates}; end
if ~iscell(subject); subject = {subject}; end
load(prc.pathFinder('expList'), 'expList');

%

%Get list of available experiments for selected subject, update the paths, convert dates to datenums
availableExps = expList(strcmp({expList.subject}', subject) & strcmp({expList.expDef}', expDef));
availableExps = prc.updatePaths(availableExps);
if isempty(availableExps); warning(['No processed files matching ' subject{1}]); return; end
availableDateNums = datenum(cell2mat({availableExps.expDate}'), 'yyyy-mm-dd');

%Depending on the "requestedDates" input, filter the available datnums
selectedDateNums = cell(size(requestedDates,1),1);
for i = 1:size(requestedDates,1)
    currDat = requestedDates{i};
    if strcmpi(currDat(1:3), 'las')
        if numel(currDat)==4; currDat = [currDat '1']; end %#ok<*AGROW>
        lastDate = str2double(currDat(5:end));
        selectedDateNums{i} = availableDateNums(end-min([lastDate length(availableDateNums)])+1:end);
    elseif strcmpi(currDat(1:3), 'fir')
        if numel(currDat)==5; currDat = [currDat '1']; end
        lastDate = str2double(currDat(6:end));
        selectedDateNums{i} = availableDateNums(1:min([length(availableDateNums), lastDate]));
    elseif strcmpi(currDat(1:3), 'yes');  selectedDateNums{i} = availableDateNums(end-1);
    elseif strcmpi(currDat(1:3), 'all');  selectedDateNums{i} = availableDateNums;
    elseif contains(lower(currDat), ':')
        dateNums = datenum(strsplit(currDat, ':')', 'yyyy-mm-dd');
        selectedDateNums{i} = availableDateNums(availableDateNums>=dateNums(1) & availableDateNums<=dateNums(2));
    else, selectedDateNums = datenum(requestedDates, 'yyyy-mm-dd');
    end
end
if iscell(selectedDateNums); selectedDateNums = unique(cell2mat(selectedDateNums)); end

%Get selected paths to load based on the selected dates. Check if blk and raw data exist in processed .mat files (based on whoD variable)
selectedFiles = {availableExps(ismember(availableDateNums, selectedDateNums)).processedData}';
if isempty(selectedFiles); warning(['No processed files matching ' subject{1} ' for requested dates']); return; end
whoD = cellfun(@(x) load(x, 'whoD'), selectedFiles, 'uni', 0);
whoD = [whoD{:}]'; whoD = {whoD(:).whoD}';
if any(~cellfun(@(x) contains('blk', x), whoD)); error('No blk for one or more requested files'); end
if any(~cellfun(@(x) contains('raw', x), whoD)); error('No raw for one or more requested files'); end

%Load the blk variables from the requested dates and concatenate them into a structure array
blk = cellfun(@(x) load(x, 'blk'), selectedFiles, 'uni', 0);
blk = [blk{:}]'; blk = [blk(:).blk]';

%If raw data is requested, load it, and add it to the concatenated blocks (in a "raw" field)
if contains(lower(extraData), {'raw'; 'all'})
    raw = cellfun(@(x) load(x, 'raw'), selectedFiles, 'uni', 0);
    for i = 1:length(raw); blk(i).raw = raw{i}.raw; end
end

%If eph data is requested, load the "eph" (spikes) and "tim" (timeline) data and add it to the concatenated blocks (as "tim" and "eph" fields
if contains(lower(extraData), {'eph'; 'all';'ephTmp'})
    timelineAvailable = find(cellfun(@(x) contains('tim', x), whoD));
    timeline = cellfun(@(x) load(x, 'tim'), selectedFiles(timelineAvailable), 'uni', 0);
    if ~isempty(timeline)
        timeline = [timeline{:}]'; timeline = [timeline(:).tim]';
        for i = 1:length(timelineAvailable); blk(timelineAvailable(i)).timeline = timeline(i); end
    end
    
    if strcmpi(extraData, 'ephTmp'); eTag = 'ephTmp'; else, eTag = 'eph'; end
    ephysAvailable = find(cellfun(@(x) contains(eTag, x), whoD));
    ephys = cellfun(@(x) load(x, eTag), selectedFiles(ephysAvailable), 'uni', 0);
    if ~isempty(ephys)
        ephys = [ephys{:}]';
        for i = 1:length(ephysAvailable); blk(ephysAvailable(i)).ephys = ephys(i).(eTag); end
    end
end

data = blk;
end