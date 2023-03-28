function plotAllenOutlines(projTag, regions, axH, tv, st, av)
%% Load atlas
if ~exist('projTag', 'var'); projTag = 'cor'; end
if ~exist('axH', 'var'); axH = axes; end

% Plot brain to overlay probes
set(gca, 'YDir','reverse'); hold on;
axis image off;
bregma = [540,0,570];
%%
switch projTag
    case 'cor'
        pIdx = 1;
        pOrd = [2,3,1];
        av = av(1:bregma(1),:,:);
    case 'top'
        pIdx = 2;
        pOrd = [1,3,2];
    case 'sag'
        pIdx = 3;
        pOrd = [2,1,3];
end
brainOutline = cell2mat(bwboundaries(permute((max(av,[],pIdx)) > 1,pOrd), 'noholes'));
plot(axH, brainOutline(:,2), brainOutline(:,1),'k','linewidth',4);

%%
cOrd = 'rbmcgy';
for i = 1:length(regions)
    reg = regions(i);
    regPath = st.structure_id_path(find(strcmp(st.acronym,reg)));
    regIdx = find(contains(st.structure_id_path,regPath));
    regMask = ismember(av, regIdx);

    regOutline = bwboundaries(permute((max(regMask,[],pIdx))>0,pOrd), 'noholes');
    regOutline(cellfun(@length, regOutline)<100) = [];
    cellfun(@(x) plot(axH, x(:,2), x(:,1),cOrd(i),'linewidth',2), regOutline)
end

%%

% % Plot projections
% horizontal_outline = bwboundaries(permute((max(av,[],2)) > 1,[3,1,2]));
% 
% str_id = find(strcmp(st.safe_name,'Caudoputamen'));
% str_coronal_outline = bwboundaries(permute((max(av == str_id,[],1)) > 0,[2,3,1]));
% str_horizontal_outline = bwboundaries(permute((max(av == str_id,[],2)) > 0,[3,1,2]));
% 
% vstr_id = find(strcmp(st.safe_name,'Nucleus accumbens'));
% vstr_coronal_outline = bwboundaries(permute((max(av == vstr_id,[],1)) > 0,[2,3,1]));
% 
% cellfun(@(x) plot(horizontal_axes,x(:,2),x(:,1),'k','linewidth',2),horizontal_outline)
% cellfun(@(x) plot(horizontal_axes,x(:,2),x(:,1),'b','linewidth',2),str_horizontal_outline)
% axis image off;
% 
% cellfun(@(x) plot(coronal_axes,x(:,2),x(:,1),'k','linewidth',2),coronal_outline)
% cellfun(@(x) plot(coronal_axes,x(:,2),x(:,1),'b','linewidth',2),str_coronal_outline)
% cellfun(@(x) plot(coronal_axes,x(:,2),x(:,1),'color',[0.8,0,0.8],'linewidth',2),vstr_coronal_outline)
% axis image off;
% 
% %% Outline of VisAM in CCF
% 
% % TO USE THIS: open allen_ccf_npx (might also work on allenAtlasBrowser,
% % but that seems broken at the moment?), move to desired annotated slice,
% % click once, then the code below will pull that slice and draw outlines
% 
% % grab slice from gui into curr_ccf_slice
% curr_ccf_slice = get(gco,'CData');
% 
% 
% am_id = find(contains(st.safe_name,'Anteromedial visual area'));
% whitematter_id = find(contains(st.safe_name,{'callosum','commissure', ...
%     'radiation','capsule','bundle','ventricle'}));
% 
% figure; hold on;
% set(gca,'YDir','reverse');
% imagesc(curr_ccf_slice);
% 
% am_outline = bwboundaries(ismember(curr_ccf_slice,am_id));
% whitematter = bwboundaries(ismember(curr_ccf_slice,whitematter_id));
% brain_outline = bwboundaries(curr_ccf_slice > 0);
% 
% cellfun(@(x) plot(x(:,2),x(:,1),'k','linewidth',2),am_outline);
% cellfun(@(x) plot(x(:,2),x(:,1),'k','linewidth',2),whitematter);
% cellfun(@(x) plot(x(:,2),x(:,1),'k','linewidth',2),brain_outline);
% 
% axis tight image off