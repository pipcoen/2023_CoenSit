function pathOut = pathFinder(pathType, pathInfo)
%% A funciton to return various paths used in processing and anlysis. Changes in file struture should be reflected here.
% INPUTS(default values)
% pathType(required)-------------A string to indicate the requested path. Can be a cell with multiple strings.
% pathInfo('noDataGiven')--------The subject information which can (but doesn't have to) contain the following fields:
%	.subject('noDataGiven')------------------Name of the subject
%	.expDate('noDataGiven')------------------Date of the experiment
%	.expNum('noDataGiven')-------------------Number of experiment
%	.datNum('noDataGiven')-------------------Number of experiment

% OUTPUTS
% pathOut---------------------------The requested path
% directoryCheck--------------------An indicator of whether the computer has 'server' or only 'local' access. Or 'all'.

%% Check inputs, extract variables from struture, and convert all to cells
if ~exist('pathType', 'var'); error('pathType required'); end
if ~exist('pathInfo', 'var'); [pathInfo.subject, pathInfo.expDate, pathInfo.expNum, pathInfo.datNum] = deal('noDataGiven'); end
if ~isfield(pathInfo, 'subject'); subject = 'noDataGiven'; else, subject = pathInfo.subject;  end
if ~isfield(pathInfo, 'expDate'); expDate = 'noDataGiven'; else, expDate = pathInfo.expDate;  end
if ~isfield(pathInfo, 'expNum'); expNum = 'noDataGiven'; else, expNum = pathInfo.expNum;  end
if ~isfield(pathInfo, 'datNum'); datNum = 'noDataGiven'; else, datNum = pathInfo.datNum;  end

if isnumeric(expNum); expNum = num2str(expNum); end
if isnumeric(expDate); expDate =  datestr(expDate, 'yyyy-mm-dd'); end

if ~iscell(pathType); pathType = {pathType}; end
if ~iscell(subject); subject = {subject}; end
if ~iscell(expDate); expDate = {expDate}; end
if ~iscell(expNum); expNum = {expNum}; end
if ~iscell(datNum); datNum = {datNum}; end

%% Make initial directory decisions based on dates and the computer that the program is running on.
%Assign the drive name and directoryCheck depending on where Pip keeps his dropbox
dataLocation = 'D:\Dropbox (Neuropixels)\MouseData\2023_CoenSit\';

%Assign locations for the raw data, processed data etc. depending on the access of the computer being used.
pathOut = cell(size(subject,1), length(pathType));
for i = 1:size(subject,1)
    %Set up paths based on the subject, date, etc.
    processedFileName = [subject{i} '\' subject{i} '_' expDate{i}([3:4 6:7 9:10]) '_' expNum{i}  'Proc.mat'];
    
    for j = 1:length(pathType)
        switch lower(pathType{j})                                                                             %hardcoded location of...

            case 'processeddirectory'; pathOut{i,j} = dataLocation;                                     %local processed data directory
            case 'processedfolder'; pathOut{i,j} = [dataLocation subject{i}];                           %local processed data folder
            case 'processeddata'; pathOut{i,j} = [dataLocation processedFileName];                      %local processed data file

            case 'explist'; pathOut{i,j} = [dataLocation 'XSupData\expList.mat'];                                %the master list of experiments
            case 'ephysrecord'; pathOut{i,j} = [dataLocation 'XSupData\ePhysRecord.mat'];                        %an excel sheet with ephys records
            case 'ephysrecordnp2'; pathOut{i,j} = [dataLocation 'XSupData\ephysrecordNP2.mat'];                        %an excel sheet with ephys records
            case 'allenatlas'; pathOut{i,j} = [dataLocation 'XSupData\Atlas\allenCCF\']; %local allan atlas directory
            case 'probepath'; pathOut{i,j} = [dataLocation 'XHistology\' subject{i} '\probe_histIdx' expNum{i}];  %probe vectors estimated from histology
        end
    end
end
if length(pathOut) == 1; pathOut = pathOut{1}; end
