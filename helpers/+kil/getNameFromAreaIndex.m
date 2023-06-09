function areaName = getNameFromAreaIndex(areaIdx, st)

areaRef = [{...
    '/184/', 'FRP';...
    '/698/', 'OLF';...
    '/654/', 'PL';...
    '/985/', 'MOp'; ...
    '/993/', 'MOs'; ...
     '/31/', 'ACA'; ...
     '/37/', 'SSp';...
    '/669/', 'VIS';...
    '/254/', 'RSP';...
    '/302/', 'SCs';...
    '/714/', 'ORB';...
    '/672/', 'CP';...
     '/56/', 'ACB';...
   '/1022/', 'GPe';...
    '/242/', 'LS';...
     '/44/', 'ILA';...
    '/972/', 'PL';...
    '/254/', 'RSP';...
   }];


structurePath = st.structure_id_path(areaIdx);
pathRef = sum(cell2mat(arrayfun(@(x) contains(structurePath, areaRef(x,1))*x, 1:size(areaRef,1), 'uni', 0)),2);
areaName(pathRef~=0) = areaRef(pathRef(pathRef~=0),2);
areaName(pathRef==0) = deal({'Nan'});
areaName = areaName';
end