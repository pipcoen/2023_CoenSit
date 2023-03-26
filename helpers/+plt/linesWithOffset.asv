function linesWithOffset(inDat, offset, eIdx)
%% Plots connected lines with a specified offset
% INPUTS(default values)
% inDat(required)-[nxm] data to plot n lines across m points
% offset----------[nx1] the offset to apply to each row


nXPnts = size(inDat,2);
inDat = inDat - repmat(offset, 1, nXPnts);
yDat = cell2mat(arrayfun(@(x) [inDat(~eIdx,x); inDat(eIdx,x); mean(inDat(:,x))], 1:nXPnts, 'uni', 0));
xDat = cell2mat(arrayfun(@(x) yDat(:,1)*0+x-0.5, 1:nXPnts, 'uni', 0));

set(axH, 'position', get(axH, 'position').*[1 1 (0.2*nXPnts) 1]);
hold on
for i = 1:nXPnts-1
    cellfun(@(x,y) plot(x,y, 'k','HandleVisibility','off'), num2cell(xDat(:,i:i+1),2), num2cell(yDat(:,i:i+1),2));
end

xDatN = xDat(1:end-2,:);
yDatN = yDat(1:end-2,:);
plot(xDatN, yDatN,'ok', 'MarkerEdgeColor', 'k','MarkerFaceColor', 'k', 'MarkerSize',5);
plot(xDat(end-1,:), yDat(end-1,:),'sc', 'MarkerEdgeColor', 'c','MarkerFaceColor', 'c', 'MarkerSize',6);
plot(xDat(end,:), yDat(end,:),'^c', 'MarkerEdgeColor', 'c','MarkerFaceColor', 'c', 'MarkerSize',6);

xlim([0 nXPnts]);
end