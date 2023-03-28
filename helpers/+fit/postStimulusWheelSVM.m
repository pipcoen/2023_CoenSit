function postStimulusWheelSVM
s = spatialAnalysis('all', 'behaviour', 0, 1, 'raw');
%%
maxSamples = 5000;
% Set the time window for the SVM analysis
difTim = 0.005;
velShift = 0.01;
svmTimes = 0.04:difTim:0.3;
%%
for k = 1:2
    % Loop over the blocks in the experiment
    for i  = 1:length(s.blks)
        % Loop over the experiments in the block
        subBlk = spatialAnalysis.getBlockType(s.blks(i),'norm',1);
        % Determine the stimulus type and set the color for plotting
        if k == 1
            stimFilt = subBlk.tri.trialType.visual;
            cCol = 'm';
        else
            stimFilt = subBlk.tri.trialType.auditory;
            cCol = 'c';
        end
        subBlk = prc.filtBlock(subBlk, stimFilt);

        for j = 1:s.blks(i).tot.experiments
            fprintf('Subject %d of %d, Experiment %d of %d \n', i, length(s.blks), j, s.blks(i).tot.experiments);

            % Filter data to include only the current experiment
            blk = prc.filtBlock(subBlk, subBlk.tri.expRef==j);

            % Select a random subset of trials with equal choices
            selectIdx = prc.makeFreqUniform(blk.tri.stim.visDiff > 0 | blk.tri.stim.audDiff > 0, 1, [], maxSamples);
            blk = prc.filtBlock(blk, selectIdx);
            
            % Get the response and reaction time data for the trials
            testPrm = blk.tri.stim.visDiff > 0 | blk.tri.stim.audDiff > 0;
            
            % Compute the instantaneous velocity of the wheel for each trial
            rawWheelTV = blk.tri.raw.wheelTimeValue;
            wheelPos = cellfun(@(x) double(interp1(x(:,1), x(:,2), svmTimes, 'pchip', 'extrap')), rawWheelTV, 'uni', 0);
            wheelVelInst = cellfun(@(x) x((2+1):end) - x(1:end-2), wheelPos, 'uni', 0);
            wheelVelInst = num2cell(cell2mat(wheelVelInst),1);
            
            % Train an SVM model for each trial and store the performance
            svmMod.reactDir{i,j,k} = testPrm;
            modResult = cellfun(@(x) fitcsvm(x,testPrm, 'CrossVal', 'on', 'KFold', 2), wheelVelInst, 'uni', 0);
            svmMod.modPerf{i}(j,:,k) = cell2mat(cellfun(@(x) mean(x.Y==x.kfoldPredict), modResult, 'uni', 0));
        end
        svmMod.subject{i} = blk.exp.subject{1};
        svmMod.svmTimes = (svmTimes(1)+velShift):difTim:(svmTimes(end));
        
        hold on;
        plot(svmMod.svmTimes,mean(svmMod.modPerf{i}(:,:,k)), cCol);
        drawnow;
        disp(['Done: ' blk.exp.subject{1}]);
    end
end
saveDir = [prc.pathFinder('processeddirectory') '\XSupData\'];
save([saveDir 'figS1gSVMModelData.mat'], 'svmMod')
end

