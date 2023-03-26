function structWithoutFields = chkThenRemoveFields(inputStructure, fields2Remove)
%% A simple function that checks whether a field exists in a structure, and then removes it. Useful before rmfield errors if the field doesn't exist

fieldNames = fields(inputStructure);
structWithoutFields = rmfield(inputStructure, fieldNames(ismember(fieldNames, fields2Remove)));
end