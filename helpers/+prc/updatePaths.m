function expList = updatePaths(expList)
%% Function that adds a bunch of paths to the expList. Doing this separately ensures paths are up to date, and makes expList easier to browse 
% INPUTS(default values)
% expList(load expList)--The current expList (or any structure with fields: "subject", "expDate" and "expNum" 

% OUTPUTS
% expList----------------The updated expList with new fields (all paths) as follows:
%	.processedData-----------local processed data file (can be same as below if no local copy)

%%
%Load expList from if no input is given
if ~exist('expList', 'var'); expList = load(prc.pathFinder('expList'), 'expList'); expList = expList.expList; end

%Collect all paths. paths2Add is for inputs to pathfinder, which differ slights from the fields of expList (which are defined by fieldNames)
paths2Add = {'processedData'};
fieldNames = {'processedData'};
dateNums = num2cell(deal(dtstr2dtnummx({expList.expDate}', 'yyyy-MM-dd')));

%Create pathInfo inputs. Using a cell like this (with the datenums already calculated) dramatically increases speed.
pathInfo.subject = {expList.subject}';
pathInfo.expDate = {expList.expDate}';
pathInfo.expNum = {expList.expNum}';
pathInfo.datNum = dateNums;

%Change the names of expList fields to match "fieldNames"
newPaths = prc.pathFinder(paths2Add, pathInfo);
for i = 1:length(expList)
    for j = 1:length(fieldNames)
        expList(i).(fieldNames{j}) = newPaths{i,j};
    end
end
