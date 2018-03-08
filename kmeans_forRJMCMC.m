clearvars

stadir = 'Data\Projects\SEGMeNT_4\AF\MBEY\';

load([stadir 'PreCalculated_RFs_Ps_1_100s_'...
    'None_Accardo2017Expanded_Accardo2017Expanded_19.10.17.mat'])
load([stadir 'Migration_Models.mat']);




%%
% First, copy in the good_models output 
close all
% the standard depths array for the inversion with each interval halved
deps = [0:0.1:10, 10.5:0.5:60, 62.5:2.5:195]; 

cmap=[83 94 173; 165 186 232; 193 226 247;213 227 148; ...
    233 229 48; 229 150 37; 200 30 33]./255;


for k=1:2
    figure('position', [2, 42+300*(k-1), 1277, 300], 'color','w'); 
    axes('position',[0.05, 0.15, 0.095, 0.8])
    plot(good_models,deps); axis ij; xlim([2, 5.5]); ylim([0, 60+130*(k-1)])
    xlabel('Vs (km/s)'); ylabel('Depth (km)');
    cmap = hsv(5+1);
    for nk = 1:5    
        rng(10)
       [icl, ccl] = kmeans(good_models',nk+1);
       axes('position',[0.05+0.14*nk, 0.15, 0.095, 0.8])
       h=plot(ccl,deps); axis ij; xlim([2,5.5]); ylim([0, 60+130*(k-1)]);
       xlabel('Vs (km/s)'); ylabel('Depth (km)');
       %set(h, {'color'}, num2cell(cmap,2));
       axes('position',[0.86, 0.9-.15*nk, 0.1, 0.12],'visible','off'); hold on
       alli = 1:length(icl);
       for nnk = 1:nk+1
           inds = find(icl == nnk);
           plot(alli(inds),icl(inds),'o','markersize',1,...
               'markerfacecolor',cmap(nnk,:)); 
           xlim([0, length(icl)]); caxis([1,nk+1]); ylim([0, nk+2])
       end
    end
    axes('position',[0,0,1,1],'visible','off'); hold on
    text(0.865,0.1,'Increasing iterations -->');
    text(0.965,0.825,'nk: 2');
    text(0.965,0.675,'nk: 3');
    text(0.965,0.525,'nk: 4');
    text(0.965,0.375,'nk: 5');
    text(0.965,0.225,'nk: 6');

    
end