%% This scripts generates figure s7f (left part) of the paper.

addpath(genpath('/Users/timothysit/multisensory-integration/matlab_src'));


weight_names = {"movRightKernelMean"};
kernel_names = {'choose right'}; %{'aud', 'vis', 'choose left', 'choose right'};
session_types = {'active'}; % {'active', 'passive'};

%% Load regression results 

passive_regression_results = readtable('passive_regression_w_positions.csv');
active_regression_results = readtable('active_regression_w_positions.csv');
use_mean_per_pen = 1;
min_neuron = 30;


% hemisphere can be 'l', 'r', 'c' (combined)

for hemisphere = ['c']

    for session_type_idx = 1:length(session_types)
        session_type = session_types{session_type_idx};
        if strcmp(session_type, 'active')
            regression_results = active_regression_results;
            

            % RGB 
            %weight_colors = [0 0.4470 0.7410; 1 0.647, 0; ...
            %                   0 0 1; 0.54 0 0];
            
            % HSV 
            weight_colors = [... 
               %  272, 0, 100; % purple 
                % 39, 0, 100; % orange 
                % 228, 0, 100; % blue 
               0, 0, 100;   % red 
            ];

        else 
            regression_results = passive_regression_results;
            % weight_colors = [0 0.4470 0.7410; 1 0.647, 0];
            % HSV
            weight_colors = [... 
                272, 0, 100; % purple 
                % 39, 0, 100; % orange  
            ];
        end 
        
        for n_weight = 1:length(weight_names)
            weights = regression_results.(weight_names{n_weight});
            alpha = (weights - min(weights)) / (max(weights) - min(weights));
    
            AP = regression_results.AP;
            DV = regression_results.DV;
            ML = regression_results.ML;
            cellLocations = [AP, DV, ML];
            colorName = weight_colors(n_weight, :); % [0 0.4470 0.7410];
            figPath = sprintf('%s_%s_%s_hemisphere.pdf', session_type, weight_names{n_weight}, hemisphere);
            title_txt = sprintf('%s hemisphere, %s kernel', hemisphere, kernel_names{n_weight});
            if use_mean_per_pen == 1
                penRefPerCell = regression_results.penRef;
                uniquePen = unique(penRefPerCell);
                numPen = length(unique(penRefPerCell));
                penAvecellLocations = zeros(numPen, 3);
                penAveWeights = zeros(numPen, 1);
                penNumCell = zeros(numPen, 1);
                for penIdx = 1:numPen 
                    penRef = uniquePen(penIdx);
                    subsetIdx = find(penRefPerCell == penRef);
                    penAvecellLocations(penIdx, :) = mean(cellLocations(subsetIdx, :), 1);
                    penAveWeights(penIdx) = mean(weights(subsetIdx));
                    penNumCell(penIdx) = length(subsetIdx);
                end
                
                penAveWeights = penAveWeights(find(penNumCell >= min_neuron));
                penAvecellLocations = penAvecellLocations(find(penNumCell >= min_neuron), :);
    
                alpha = (penAveWeights - min(penAveWeights)) / (max(penAveWeights) - min(penAveWeights));
    
                plotCellsOn2DBrain(hemisphere, penAvecellLocations, colorName, alpha, title_txt, weight_names{n_weight}, figPath)
    
            else
    
                plotCellsOn2DBrain(hemisphere, cellLocations, colorName, alpha, title_txt, weight_names{n_weight}, figPath)
                

            end

            % close all
        end 
       
    end 
end 
