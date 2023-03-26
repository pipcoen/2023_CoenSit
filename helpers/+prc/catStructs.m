function combinedStruct = catStructs(structArray, recurrent)
%% Function take a structure array and combine the fields to give a single structure with concatenated fields

%INPUTS(default values)
%subject(structArray)----------An array of structures

%OUTPUTS
%combinedStruct----------------A single stuct with the original structures combined (concatenated across dim 1)

%%
%Check with fields are in the structure
structFields = fields(structArray);
if ~exist('recurrent', 'var'); recurrent = 0; end
%For iterate over each field, and combine the information within
for i = 1:length(structFields)
    if ~any(cellfun(@isstruct, {structArray.(structFields{i})}'))
        %If the field is not a structArray...
        if ischar(structArray(1).(structFields{i}))
            %If the field is a string, concatenate the fields as a cell array of strings
            tDat = {structArray(:).(structFields{i})};
            combinedStruct.(structFields{i}) = tDat(:);
        else
            %If not a string (i.e. cells, vectors, etc.), then just concatenate using vertcat
            combinedStruct.(structFields{i}) = vertcat(structArray(:).(structFields{i}));
        end
    else
        %If the field is itself a struture array, use catStructs itteratively to combine it
        combinedStruct.(structFields{i}) = prc.catStructs(vertcat(structArray.(structFields{i})),1);
    end
end

if isfield(combinedStruct, 'tot') && ~recurrent
    for fld = fields(combinedStruct.tot)' 
        combinedStruct.tot.(fld{1}) = sum(combinedStruct.tot.(fld{1}));
    end
end
end
