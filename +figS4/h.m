function h
%% This function plots a panel from the manuscript (Figure S4h)
% First do the stats
loadDir = [prc.pathFinder('processeddirectory') 'XSupData\'];
load([loadDir 'figS4hInactCompResults'], 'trainTestGroups', ...
    'testGrp1LogLik', 'testGrp2LogLik', 'normEstRepeats');

logLikR = zeros(5,5);
nReg = 1:5;
selfTestlogLikR = zeros(4,1);
for i = nReg
    for j = nReg
        if i == j
            compIdx = nReg(~ismember(nReg,j));
            for k = 1:4
                % Use all 4 instances of self-test to estimate SelfLLR
                idx = ismember(trainTestGroups, [i,compIdx(k)], 'rows');
                selfTestlogLikR(k,1) = mean(testGrp1LogLik{idx}(1:normEstRepeats));
            end
            logLikR(i,j) = mean(selfTestlogLikR);
            continue; 
        end
        idx = ismember(trainTestGroups, [i,j], 'rows');
        sIdx = normEstRepeats+1:length(testGrp1LogLik{idx});
        
        LLRSelf = mean(testGrp1LogLik{idx}(1:normEstRepeats));
        LLRTest = mean(testGrp2LogLik{idx}(1:normEstRepeats));       
        shuffTest = sort([(LLRSelf-LLRTest); testGrp1LogLik{idx}(sIdx)-testGrp2LogLik{idx}(sIdx)]);
        
        pVal(i,j) = find(shuffTest == LLRSelf-LLRTest)/length(sIdx)*5;        
        logLikR(i,j) = LLRTest;
    end
end

%%
figure;

cCol = {[1, 0.65, 0]; [255, 215, 0]/255; 'm'; 'b'; 'k'};
for i = nReg
    for j = nReg
        plot(logLikR(i,j)*-1,i, '.', 'Color', cCol{j}, 'MarkerSize',25)
        hold on;
    end
end
box off;
xlim([-1, -0.55])
ylim([0.5, 5.5])
set(gca, 'YTick', 1:5, 'YTickLabel', {'Frontal'; 'Visual'; 'Lateral'; 'Somat.'; 'Control'}, 'YDir', 'reverse');
set(gca, 'XTick', [-1, -0.55]);
set(gca, 'Position', get(gca, 'Position').*[1.2 1.2 0.8 1]);
xlabel('Loglikelihood')
ylabel('Testing Region')

f=get(gca,'Children');
h = legend([f(1), f(2), f(3), f(4), f(5)],'Frontal', 'Visual', 'Lateral', 'Somat.', 'Control');
set(h, 'box', 'off')
h.Position = h.Position.*[1.4,1,1,1];
title(h, 'Training region');
end