classdef spatialAnalysis < matlab.mixin.Copyable
    %% SpatialAnalysis object that extracts beahvioral information for a specified animal or set of animals. The resulting object has a set of methods
    %that can be used to plot various aspects of animal behavior. NOTE: This function is designed to operate on the output of convertExpFiles and
    %most methods are specific to multisensoty spatial integration plots.
    
    %INPUTS(default values)
    %subjects(required)------------A cell array of subjects collect parameter and block files for
    %expDate('last')---------------A cell array of dates (case insensitive), one for all subjects or one for each subject
    %        'yyyy-mm-dd'--------------A specific date
    %        'all'---------------------All data
    %        'lastx'-------------------The last x days of data (especially useful during training)
    %        'firstx'------------------The first x days of date
    %        'yest'--------------------The x-1 day, where x is the most recent day
    %        'yyyy-mm-dd:yyyy-mm-dd'---Load dates in this range (including the boundaries)
    %        A dataTag (see prc.keyDates)
    %combineMice(0)----------------A tag to indicate whether to combine multiple mice, or split a single mouse into separate days
    %        0-------------------------Individual mice are returned as individual structures
    %        1-------------------------Mice combined into a single uber-mouse with a concatenated name
    %        -1------------------------Only used with one mouse, and means that the mouse is split into individual sessions
    %modalParams(0)----------------Indicates whether to eliminate days where the pamater set (aud/vis combos and expType) didn't match the mode
    %extraTag('eph')---------------Indicates whether you want to load extra data
    %        'eph'---------------------load the ephys data, if available, for each session date
    %        'raw'---------------------load the raw data, (rarely needed wheel times from signals etc.)
    %expDef('multiSpaceWorld')-----Specify the particular expDef to load
    
    properties (Access=public)
        blks;                  %Block files loaded for each subject (one cell per subject)
        glmFit;                %GLM class for post-fitting of data in blks
        hand;                  %Handles to current axis/figure being used for plotting
    end
    
    %%METHODS (there are other methods contained in separate .m files)
    methods
        %% Central function that loads the requested mouse data into a spatial analysis object
        function obj = spatialAnalysis(subjects, expDate, combineMode, modalParams, extraTag, expDef)
            
            %Initialize fields with default values if no vaules are provided.
            if ~exist('subjects', 'var'); error('Must specify which subject to load data from'); end
            if ~exist('expDate', 'var'); expDate = {'last'}; end
            if ~exist('combineMode', 'var'); combineMode = 0; end
            if ~exist('modalParams', 'var'); modalParams = 0; end
            if ~exist('extraTag', 'var'); extraTag = 'eph'; end
            if ~exist('expDef', 'var'); expDef = 'multiSpaceWorld'; end
            
            %Make sure that all fields are cells. If not, convert to cells. If "all" mice are requested, create cell array of all mice.
            if ~iscell(expDate); expDate = {expDate}; end
            if ~iscell(subjects); subjects = {subjects}; end
            if any(strcmp(subjects, 'all'))
                subjects = [arrayfun(@(x)['PC0' num2str(x)], 10:99,'uni', 0),'DJ006', 'DJ007','DJ008','DJ010'];
            end
            
            %If there is only one date provided, repeat for all subjects.
            if length(expDate) < length(subjects); expDate = repmat(expDate, length(subjects),1); end
            subjects = subjects(:); expDate = expDate(:);  %Make sure subjects and rows are columns.
            
            %If a keyDates tag was used (e.g. "behaviour") instead of a "real" date, "prc.keyDates" will get the corresponding date range. If a tag
            %was not used, then the "expDate" input will not match a data tag in prc.keyDates, and so will be unchanged.
            expDate = arrayfun(@(x,y) prc.keyDates(x,y), subjects(:), expDate(:), 'uni', 0);
            subjects = subjects(~cellfun(@isempty, expDate)); %Removes excess subjects in the 'all' case
            expDate = expDate(~cellfun(@isempty, expDate));   %Removes corresponding dates
            
            %Run 'changeMouse' function, which does all the loading and combining of the data
            obj = changeMouse(obj, subjects, expDate, combineMode, modalParams, expDef, extraTag);
        end
        
        %% This function uses the curated inputs to actually load and combine the data as requested
        function obj = changeMouse(obj, subjects, expDate, combineMode, modalParams, extraTag, expDef)
            
            %INPUTS are defined above, but some defaults be redefined here in case this method is called for an existing object
            if ~exist('combineMode', 'var'); combineMode = 0; end
            if ~exist('extraTag', 'var'); extraTag = 'none'; end
            if ~exist('expDef', 'var'); expDef = 'multiSpaceWorld'; end
            
            %Load the data for the requested subjects/dates using prc.getDataFromDates. Concatenate into one structure array, "blks"
            obj.blks  = cellfun(@(x,y) prc.getDataFromDates(x, y, extraTag, expDef), subjects(:), expDate(:), 'uni', 0);
            obj.blks = vertcat(obj.blks{:}); %NOTE: currently errors if some mice have ephys data and others don't
            
            %Get list of subjects and a reference index. Modify depending on the "combineMode"
            mouseList = unique({obj.blks.subject})';
            [~, subjectRef] = ismember({obj.blks.subject}', mouseList);
            if combineMode==1
                mouseList = {cell2mat(mouseList')};
                subjectRef = subjectRef*0+1;
            elseif combineMode==-1 
                if length(mouseList)~=1; error('Request one mouse if you want it split into separate days'); end
                mouseList = arrayfun(@(x) [mouseList{1} ':' num2str(x{1})], {obj.blks.expDate}', 'uni',0);
                [obj.blks.subject] = deal(mouseList{:});
                subjectRef = (1:length(subjectRef))';
            end
            
            %This loop removes the less common sessions for each mouse if "modalParams==1" and the mouse has a mix of paramters/expTypes
            retainIdx = ones(length(obj.blks),1)>0;
            for i = 1:max(subjectRef)
                mouseConditions = {obj.blks(subjectRef==i).conditionParametersAV}';
                mouseExpTypes = {obj.blks(subjectRef==i).expType}';
                [conditionSets, ~, setIdx] = unique(cellfun(@(x,y) [num2str(x(:)'), y], mouseConditions, mouseExpTypes,'uni',0));
                if length(conditionSets)>1 && modalParams
                    fprintf('WARNING: Several parameter sets in date range for %s. Using mode\n', mouseList{i});
                    retainIdx(subjectRef==i) = retainIdx(subjectRef==i).*(setIdx == mode(setIdx));
                end
            end
            obj.blks = obj.blks(retainIdx);
            subjectRef = subjectRef(retainIdx);
            
            %Run the blk array through "combineBlocks" which converts blks into their final format
            obj.blks = cell2mat(arrayfun(@(x) prc.combineBlocks(obj.blks(subjectRef==x)), 1:size(mouseList,1), 'uni', 0)');
            obj.hand.axes = [];
            obj.hand.figure = [];
        end
    end
    
    methods (Static)
        %% Function to filter blocks based on some predefined tags
        function filteredBlk = getBlockType(blk, tag, removeTimeouts, removeRespNans)
            if ~exist('tag', 'var'); error('Must specificy tag'); end
            if ~exist('removeTimeouts', 'var'); removeTimeouts = 1; end
            if ~exist('removeRespNans', 'var'); removeRespNans = 1; end
            validTrials = blk.tri.trialType.validTrial;                                          %Trials that weren't repeats of an incorrect decision
            timeOuts = blk.tri.outcome.responseRecorded==0 | blk.tri.outcome.timeToFeedback > 1.5;   %Timout trials (timeouts, or response > 1.5s)
            laserType = blk.tri.inactivation.laserType;                                          %Lasertype used on each trial.   
            responseNans = isnan(blk.tri.outcome.responseCalc);
            removeIdx = timeOuts*0;
            if removeTimeouts; timeOuts = timeOuts*0; end                                        %If not removing timeouts, filter becomes all zeros
            if removeRespNans; removeIdx(responseNans & ~timeOuts) = 1; end                      %If not removing responseNans, filter becomes all zeros
            
            regBlk = prc.filtBlock(blk, (laserType==0 | isnan(laserType)) & ~removeIdx & validTrials, 'tri');
            lasBlk = prc.filtBlock(blk, (laserType~=0 & ~isnan(laserType)) & ~removeIdx & validTrials, 'tri');
            
            switch lower(tag(1:3))
                case 'nor'; filteredBlk = regBlk;
                case 'las'; filteredBlk = lasBlk;
            end
        end
        
        function alterFigure(currentFigureHandle, ~)
            obj = get(currentFigureHandle.Number, 'UserData');
            switch lower(get(currentFigureHandle, 'Tag'))
                case 'boxres'
                    obj.viewBoxPlots('num', currentFigureHandle);
                case 'boxnum'
                    obj.viewBoxPlots('res', currentFigureHandle);
            end
        end
        
        function alterAxes(currentAxesHandle, ~)
            obj = get(currentAxesHandle.Number, 'UserData');
            switch lower(get(currentAxesHandle.Number, 'Tag'))
                case 'boxres'
                    obj.viewBoxPlots('num', currentAxesHandle);
            end
        end
    end
end