function postStimulusWheelSVM
s = spatialAnalysis('all', 'behaviour', 0, 1, '');
nSamples = 5000;
for k = 1:2
    for i  = 1:length(obj.blks)
        for j = 1:obj.blks(i).tot.experiments
            blk = spatialAnalysis.getBlockType(s.blks(i),'norm',1);
            blk = prc.filtBlock(blk, blk.tri.expRef==j);
            if k == 1
                stimFilt = blk.tri.trialType.visual;
                cCol = 'm';
            else
                stimFilt = blk.tri.trialType.auditory;
                cCol = 'c';
            end
            blk = prc.filtBlock(blk, stimFilt);
            
            if contains(modTag, 'mov')
                testPrm = blk.tri.outcome.responseCalc;
            elseif contains(modTag, 'stm')
                testPrm = blk.tri.stim.visDiff > 0 | blk.tri.stim.audDiff > 0;
            end
            
            selectIdx = prc.makeFreqUniform(testPrm, 1, [], nSamples);
            blk = prc.filtBlock(blk, selectIdx);
            
            difTim = 0.005;
            if contains(modTag, 'mov')
                testPrm = blk.tri.outcome.responseCalc;
                testTim = blk.tri.outcome.reactionTime;
                svmTimes = -0.06:difTim:0.1;
            elseif contains(modTag, 'stm')
                testPrm = blk.tri.stim.visDiff > 0 | blk.tri.stim.audDiff > 0;
                testTim = blk.tri.outcome.reactionTime*0;
                svmTimes = 0.04:difTim:0.3;
            end
            
            velShift = 0.01;
            shiftN = velShift/difTim;
            timeWinWide = arrayfun(@(x) x+(svmTimes), testTim, 'uni', 0);
            rawWheelTV = blk.tri.raw.wheelTimeValue;
            wheelPos = cellfun(@(x,y) double(interp1(x(:,1), x(:,2), y, 'pchip', 'extrap')), rawWheelTV, timeWinWide, 'uni', 0);
            wheelVelInst = cellfun(@(x) x((shiftN+1):end) - x(1:end-shiftN), wheelPos, 'uni', 0);
            wheelVelInst = num2cell(cell2mat(wheelVelInst),1);
            
            svmMod.reactDir{i,j,k} = testPrm;
            svmMod.models{i}(j,:,k) = cellfun(@(x) fitcsvm(x,testPrm, 'CrossVal', 'on', 'KFold', 2), wheelVelInst, 'uni', 0);
            svmMod.modPerf{i}(j,:,k) = cell2mat(cellfun(@(x) mean(x.Y==x.kfoldPredict), svmMod.models{i}(j,:,k), 'uni', 0));
        end
        svmMod.subject{i} = blk.exp.subject{1};
        svmMod.svmTimes = (svmTimes(1)+velShift):difTim:(svmTimes(end));
        
        hold on;
        plot(svmMod.svmTimes,mean(svmMod.modPerf{i}(:,:,k)), cCol);
        drawnow;
        disp(['Done: ' blk.exp.subject{1}]);
    end
    plot(svmMod.svmTimes,mean(cell2mat(cellfun(@(x) mean(x(:,:,k)), svmMod.modPerf, 'uni', 0)')), cCol, 'linewidth',4);
end
save(['figSSVMMoveModel_' modTag], 'svmMod')
end

