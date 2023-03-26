function newBlk = filtBlock(blk, criterion, filterTag)
%% Function to fulter structure.
totFields = fields(blk.tot);
if ~exist('criterion', 'var'); filtered = []; return; end
if ~exist('filterTag', 'var')
    totals = cellfun(@(x) blk.tot.(x), totFields);
    if sum(length(criterion) == totals)~=1; error('Please specify tag. Type of filter unclear'); end
    filterTag = totFields(totals==length(criterion));
end
if iscell(filterTag); filterTag = filterTag{1}; end
filterTag = filterTag(1:3);
criterion = criterion>0;

totFields = totFields(~strcmp(totFields, 'subjects'))';
shortFields = fields(blk);
shortFields = shortFields(~strcmp(shortFields, 'tot'))';
for i = 1:length(totFields)
    blk.(shortFields{i}).([shortFields{i} 'Ref']) = (1:blk.tot.(totFields{i}))';
end

newBlk = blk;
newBlk.(filterTag) = filterStructRows(blk.(filterTag), criterion);
if isfield(newBlk, 'pen'); ephysExists = 1; else, ephysExists = 0; end
if strcmpi(filterTag, 'exp')
    newBlk = filterByReference(newBlk, 'expRef', newBlk.exp.expRef);
    newBlk = filterByReference(newBlk, 'subjectRef', unique(newBlk.exp.subjectRef));  
    if ephysExists
        newBlk = filterByReference(newBlk, 'penetrationRef', newBlk.pen.penRef); 
    end
    newBlk = updateTotals(newBlk, totFields, shortFields);
end

if strcmpi(filterTag, 'tri')
    newBlk.exp.numOfTrials = accumarray(newBlk.tri.expRef,1);
    newBlk = updateTotals(newBlk, totFields, shortFields);
    if any(~newBlk.exp.numOfTrials); newBlk = prc.filtBlock(newBlk, newBlk.exp.numOfTrials>0, 'exp'); end
end

if strcmpi(filterTag, 'pen')
    newBlk = filterByReference(newBlk, 'expRef', unique(newBlk.pen.expRef));
    newBlk = filterByReference(newBlk, 'subjectRef', unique(newBlk.exp.subjectRef));   
    newBlk = filterByReference(newBlk, 'penetrationRef', newBlk.pen.penRef);
    newBlk = updateTotals(newBlk, totFields, shortFields);
end

if strcmpi(filterTag, 'clu')
    newBlk.pen.numOfClusters = accumarray(newBlk.clu.penetrationRef,1);
    newBlk = updateTotals(newBlk, totFields, shortFields);
    if any(~newBlk.pen.numOfClusters); newBlk = prc.filtBlock(newBlk, newBlk.pen.numOfClusters>0, 'pen'); end
end
end

function newBlk = updateTotals(newBlk, totFields, shortFields)
newBlk.tot.subjects = length(unique(newBlk.exp.subjectRef));
for i = 1:length(totFields)
    newBlk.tot.(totFields{i}) = length(newBlk.(shortFields{i}).([shortFields{i} 'Ref']));
    newBlk.(shortFields{i}) = rmfield(newBlk.(shortFields{i}), ([shortFields{i} 'Ref']));
end
end

function filtered = filterStructRows(unfiltered, criterion)
filtered = unfiltered;
fieldNames = fields(unfiltered);
for fieldName = fieldNames'
    if isstruct(unfiltered.(fieldName{1}))
        filtered.(fieldName{1}) = filterStructRows(unfiltered.(fieldName{1}), criterion);
    else, filtered.(fieldName{1}) = unfiltered.(fieldName{1})(criterion,:,:);
    end
end
end

function filtered = filterByReference(unfiltered, refName, remainingValues)
filtFields = fields(unfiltered)';
for i = filtFields(cellfun(@(x) isfield(unfiltered.(x), refName), filtFields))
    unfiltered.(i{1}) = filterStructRows(unfiltered.(i{1}), ismember(unfiltered.(i{1}).(refName), remainingValues));
    [~,unfiltered.(i{1}).(refName)] = ismember(unfiltered.(i{1}).(refName), remainingValues);
end
filtered = unfiltered;
end