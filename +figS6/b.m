function b
%% This function plots a panel from the manuscript (Figure S6b)
lfpEx = 28;
figure;
set(gcf, 'Position', get(gcf, 'Position').*[1,1,0.5,1])
powerSpectra = flipud(log(s.blks.pen.lfpPowerSpectra{lfpEx}.powerSpectra(1:100,:)'));
freqPoints = s.blks.pen.lfpPowerSpectra{lfpEx}.freqPoints(1:100);

refChannels = [5; 44; 81; 120; 157; 196; 233; 272; 309; 348];
chan2Plot = setdiff(1:384, refChannels);

powerSpectra = bsxfun(@minus, powerSpectra, median(powerSpectra));
imagesc(freqPoints,1:length(chan2Plot), powerSpectra(chan2Plot,:));
colormap default
colorbar;
xlabel('Frequency (Hz)')
ylabel('Distance from probe tip (um)')
set(gca, 'YTick', [1, max(ylim)], 'YTickLabel', {'3840', '0'}, 'XTick', [1,100])
end