function dateRange = keyDates(subjectID, dataTag)
%% A funciton to get the "key dates" for a mouse, based on a tag for the type of data requested.
%NOTE: These are manually defined date ranges (by Pip) based on which mice were used for which experiments, and when they were used.

%INPUTS(default values)
%subjectID(required)-----The subject for which the dates are requested
%dataTag(required)-------A string representing the type of data requested. These can be...
%            'behavior'------The final version of the task without additional recording (e.g. ephys)
%            'aud5'----------Version of the task with 5 auditory locations instead of 3
%            'uniscan'-------Unilateral inactivations (with light shielding)
%            'biscan'--------Bilateral inactivations (without light shielding)
%            'm2ephys'-------M2 ephys sessions
%            'm2ephysgood'---M2 ephys sessions, but only mice with good behavior on at least one session

%OUTPUTS
%dateRange---------------The selected date range. Empty if subject doesn't match one of the subjects within a tag.

%%
switch lower(dataTag{1})
    
        case {'learning'}
        switch subjectID{1}
            case 'PC022'; dateRange = {'first35'}; 
            case 'PC027'; dateRange = {'first35'}; 
            case 'PC029'; dateRange = {'first35'}; 
            case 'PC030'; dateRange = {'first35'}; 
            case 'PC031'; dateRange = {'first35'}; 
            case 'PC032'; dateRange = {'first35'}; 
            case 'PC034'; dateRange = {'first35'}; 
            case 'PC043'; dateRange = {'first35'}; 
            case 'PC045'; dateRange = {'first35'}; 
            case 'PC046'; dateRange = {'first35'}; 
            case 'PC050'; dateRange = {'first35'}; 
            case 'PC051'; dateRange = {'first35'}; 
%             case 'DJ010'; dateRange = {'first35'}; %missing data before 208-05-22...???
            otherwise, dateRange = [];
        end
        
    %These are dates to use for behavioral analysis. Some are dates when mice where inactivated, and in this case, the inactivation trials should be
    %removed before doing further analysis.
    case {'behavior'; 'behaviour'}
        switch subjectID{1}
%             case 'PC011'; dateRange = {'2017-06-14:2017-08-16'};                            %No inactivation
%             case 'PC012'; dateRange = {'2017-09-06:2017-10-17'};                            %Regular trials within inactivation
%             case 'PC013'; dateRange = {'2017-09-05:2017-10-17'};                            %Regular trials within inactivation
%             case 'PC015'; dateRange = {'2017-09-25:2017-10-17'};                            %No inactivation
            case 'PC022'; dateRange = {'2018-02-20:2018-05-28'; '2018-05-31:2018-07-30'};   %Regular trials within inactivation
            case 'PC027'; dateRange = {'2018-02-05:2018-04-12'; '2018-04-14:2018-04-26';... %Regular trials within inactivation
                    '2018-04-30:2018-05-15'};                          
            case 'PC029'; dateRange = {'2018-02-27:2018-04-13'; '2018-04-15:2018-05-20'};   %Regular trials within inactivation
            case 'PC030'; dateRange = {'2019-03-08:2019-03-19'; '2019-03-22:2019-04-30'};   %No inactivation
            case 'PC031'; dateRange = {'2019-02-19:2019-03-01'};                            %No inactivation
            case 'PC032'; dateRange = {'2019-02-12:2019-04-02'};                            %No inactivation
            case 'PC033'; dateRange = {'2019-02-27:2019-03-26'};                            %No inactivation
            case 'PC034'; dateRange = {'2019-02-19:2019-03-11'};                            %No inactivation (Issues with eye after this point)
            case 'PC043'; dateRange = {'2019-03-06:2019-04-30'};                            %No inactivation 
            case 'PC045'; dateRange = {'2019-07-30:2019-08-26'};                            %No inactivation 
            case 'PC046'; dateRange = {'2019-07-24:2019-08-26'};                            %No inactivation
            case 'PC050'; dateRange = {'2019-07-24:2019-09-09'};                            %No inactivation
            case 'PC051'; dateRange = {'2019-07-24:2019-09-09'};                            %No inactivation
            case 'DJ006'; dateRange = {'2018-07-04:2018-08-28'; '2018-08-30:2018-09-17'};   %Regular trials within inactivation
            case 'DJ007'; dateRange = {'2018-06-20:2018-11-04'};                            %Regular trials within inactivation
            case 'DJ008'; dateRange = {'2018-05-11:2018-05-30'};                            %Regular trials within inactivation
            case 'DJ010'; dateRange = {'2018-06-20:2018-07-27'};                            %Regular trials within inactivation
            otherwise, dateRange = [];
        end
        
    case 'aud5'
        switch subjectID{1}
            case 'PC013'; dateRange = {'2017-10-18:2017-11-04'}; %Regular trials within inactivation
            otherwise, dateRange = [];
        end
        
    case 'uniscan'
        switch subjectID{1}
%             case 'PC022'; dattwo eRange = {'2018-02-20:2018-05-28'};
            case 'PC027'; dateRange = {'2018-02-05:2018-03-21'}; %Power was only 1.5mW
            case 'PC029'; dateRange = {'2018-06-12:2018-07-17'};
            case 'DJ008'; dateRange = {'2018-06-12:2018-07-17'};
            case 'DJ006'; dateRange = {'2018-08-06:2018-08-28'; '2018-08-30:2018-09-16'};
            case 'DJ007'; dateRange = {'2018-08-06:2018-10-18'};
            otherwise, dateRange = []; 
        end
        
    case 'biscan'
        switch subjectID{1}
            case 'PC010'; dateRange = {'2017-06-20:2017-07-04'}; %Power was only 1.5mW
            case 'PC012'; dateRange = {'2017-06-21:2017-07-04'}; %Power was only 1.5mW
            case 'PC013'; dateRange = {'2017-06-20:2017-07-04'}; %Power was only 1.5mW
            otherwise, dateRange = []; 
        end

    case 'biscannorm'
        switch subjectID{1}
            case 'PC010'; dateRange = {'2017-07-05:2017-07-09'}; %Power was only 1.5mW
            case 'PC012'; dateRange = {'2017-07-05:2017-07-09'}; %Power was only 1.5mW
            case 'PC013'; dateRange = {'2017-07-05:2017-07-09'}; %Power was only 1.5mW
            otherwise, dateRange = [];
        end
 
    case 'variscan'
        switch subjectID{1}
            case 'PC022'; dateRange = {'2018-05-31:2018-06-24'; '2018-07-03:2018-07-30'}; %Power was only 1.5mW
            case 'PC027'; dateRange = {'2018-06-13:2018-06-24'; '2018-07-03:2018-09-23'};
            case 'PC029'; dateRange = {'2018-07-18:2018-07-30'; '2018-08-01:2018-09-23'};
            case 'DJ006'; dateRange = {'2018-07-11:2018-08-05'};
            case 'DJ008'; dateRange = {'2018-07-18:2018-11-03'};
            case 'DJ007'; dateRange = {'2018-07-03:2018-08-05'};
            case 'DJ010'; dateRange = {'2018-07-03:2018-07-29'};
            otherwise, dateRange = [];
        end
        
    case 'm2ephys'
        switch subjectID{1}
            case 'DJ007'; dateRange = {'2018-11-28:2018-12-02'}; 
            case 'PC029'; dateRange = {'2018-10-18:2018-10-19'};
            case 'PC030'; dateRange = {'2019-05-07:2019-05-13'};
            case 'PC032'; dateRange = {'2019-04-03:2019-04-14'};
            case 'PC033'; dateRange = {'2019-03-27:2019-03-31'};
            case 'PC043'; dateRange = {'2019-05-07:2019-05-16'};
            case 'PC045'; dateRange = {'2019-08-27:2019-09-06'};
            case 'PC046'; dateRange = {'2019-08-27:2019-09-06'};
            case 'PC048'; dateRange = {'2019-09-17:2019-09-26'};
            case 'PC050'; dateRange = {'2019-09-17:2019-09-26'};
            otherwise, dateRange = [];
        end
        
    case 'm2ephysgood'
        switch subjectID{1}
            case 'PC032'; dateRange = {'2019-04-03:2019-04-14'};
            case 'PC043'; dateRange = {'2019-05-07:2019-05-16'};
            case 'PC045'; dateRange = {'2019-08-27:2019-09-06'};
            case 'PC046'; dateRange = {'2019-08-27:2019-09-06'};
            case 'PC048'; dateRange = {'2019-09-17:2019-09-26'};
            case 'PC050'; dateRange = {'2019-09-17:2019-09-26'};
            otherwise, dateRange = [];
        end
        
    case 'm2ephysmod'
        switch subjectID{1}
            case 'PC043'; dateRange = {'2019-05-07:2019-05-07'; '2019-05-09:2019-05-09'; '2019-05-14:2019-05-14'};
            case 'PC045'; dateRange = {'2019-08-27:2019-09-06'};
            case 'PC046'; dateRange = {'2019-08-27:2019-09-04'};
            case 'PC048'; dateRange = {'2019-09-17:2019-09-20'};
            case 'PC050'; dateRange = {'2019-09-17:2019-09-21'; '2019-09-24:2019-09-24'};
            otherwise, dateRange = [];
        end
        
    case 'presurg'
        switch subjectID{1}
            case 'DJ007'; dateRange = {'2018-11-13:2018-11-27'}; %Power was only 1.5mW
            case 'PC029'; dateRange = {'2018-10-03:2018-10-17'};
            case 'PC032'; dateRange = {'2019-03-13:2019-04-02'};
            case 'PC033'; dateRange = {'2019-03-12:2019-03-26'};
            case 'PC030'; dateRange = {'2019-04-21:2019-05-06'};
            case 'PC043'; dateRange = {'2019-04-26:2019-05-06'};
            case 'PC045'; dateRange = {'2019-08-20:2019-08-26'};
            case 'PC046'; dateRange = {'2019-08-20:2019-08-26'};
            case 'PC048'; dateRange = {'2019-09-11:2019-09-16'};
            case 'PC050'; dateRange = {'2019-09-11:2019-09-16'};
            otherwise, dateRange = []; 
        end
    
    case 'postsurg'
        switch subjectID{1}
            case 'DJ007'; dateRange = {'2018-11-28:2020-11-27'}; %Power was only 1.5mW
            case 'PC029'; dateRange = {'2018-10-17:2020-10-17'};
            case 'PC032'; dateRange = {'2019-04-03:2020-04-02'};
            case 'PC033'; dateRange = {'2019-03-27:2020-03-26'};
            case 'PC030'; dateRange = {'2019-05-07:2020-05-06'};
            case 'PC043'; dateRange = {'2019-05-07:2020-05-06'};
            case 'PC045'; dateRange = {'2019-08-27:2020-08-26'};
            case 'PC046'; dateRange = {'2019-08-27:2020-08-26'};
            case 'PC048'; dateRange = {'2019-09-17:2020-09-16'};
            case 'PC050'; dateRange = {'2019-09-17:2020-09-16'};
            otherwise, dateRange = [];
        end
        
    case 'modbehaviour'
        switch subjectID{1}
            case 'PC043'; dateRange = {'2019-03-06:2019-04-30'};                            %No inactivation 
            case 'PC045'; dateRange = {'2019-07-30:2019-08-26'};                            %No inactivation 
            case 'PC046'; dateRange = {'2019-07-24:2019-08-26'};                            %No inactivation
            case 'PC048'; dateRange = {'2019-09-09:2019-09-16'};                            %No inactivation
            case 'PC050'; dateRange = {'2019-07-24:2019-09-09'};                            %No inactivation
            otherwise, dateRange = [];
        end

    case 'nocentralaud'
        switch subjectID{1}
            case 'PC010'; dateRange = {'2017-05-10:2017-06-13'};                            %No inactivation
            case 'PC011'; dateRange = {'2017-05-11:2017-06-13'};                            %No inactivation
            case 'PC012'; dateRange = {'2017-05-10:2017-06-13'};                            %No inactivation
            case 'PC013'; dateRange = {'2017-05-11:2017-06-13'};                            %No inactivation
            case 'PC015'; dateRange = {'2017-05-11:2017-06-13'};                            %No inactivation
            otherwise, dateRange = [];
        end

    case 'naive'
        switch subjectID{1}
            case 'PC052'; dateRange = {'all'};                       
            case 'PC053'; dateRange = {'all'};                         
            case 'PC054'; dateRange = {'all'};                          
            case 'PC055'; dateRange = {'all'};                         
            otherwise, dateRange = [];
        end
        
    %NOTE: this is important as it allows prc.keyDates to be run on every initialization of "spatialAnalysis" because it will return the original
    %"expDate" if it doesn't match any tags. e.g. if it is a specific date, or "last2" or soemthing along these lines. 
    otherwise, dateRange = dataTag;
end
end