function p
%% This function plots a panel from the manuscript (Figure S3p)
loglikDat = fit.compareSpecifiedModels({'simpLogSplitVSplitAUnisensory_Cross5'; 'fullEmp_Cross5'});
eIdx = zeros(length(loglikDat.subjects),1);
eIdx(strcmp(loglikDat.subjects, 'PC022')) = 1;
eIdx(strcmp(loglikDat.subjects, 'PC051')) = 2;
eIdx(end) = 3;
%%

figure;
hold on
xDat = loglikDat.simpLogSplitVSplitAUnisensory_Cross5;
yDat = loglikDat.fullEmp_Cross5;
xlim([0 0.55]);
ylim([0 0.55]);
xlabel('Additive model (bits/trial)')
ylabel('Full model (bits/trial)')
axis square;
plot([0,0.55], [0,0.55], '--k')

scatter(xDat(~eIdx), yDat(~eIdx), 50, 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.5, 0.5, 0.5]);
scatter(xDat(eIdx==1), yDat(eIdx==1), 75, 'd', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.5, 0.5, 0.5]);
scatter(xDat(eIdx==2), yDat(eIdx==2), 100, '^', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.5, 0.5, 0.5]);
scatter(xDat(eIdx==3), yDat(eIdx==3), 75, 's', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.5, 0.5, 0.5]);

[~, pVal] = ttest(xDat(1:end-1), yDat(1:end-1));
text(0.4, 0.1, ['pVal ~ ' num2str(round(pVal * 100)/100)]);
end