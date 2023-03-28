function allenOutline
loadDir = [prc.pathFinder('processeddirectory') 'XSupData\'];
load([loadDir 'allenCorticalBoundaries'], 'corticalAreadBoundaries');
hold on;
bregma = [540,0,570];
for i =1:length(corticalAreadBoundaries)
    cellfun(@(x) plot((x(:,2)-bregma(3))/100, (bregma(1)-x(:,1))/100,'k'),corticalAreadBoundaries{i});
end