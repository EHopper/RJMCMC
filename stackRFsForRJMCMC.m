clearvars;% close all; clc
basedir='C:\Users\emily\OneDrive\Documents\WORK\MATLAB\Scattered_Waves\';
javaaddpath([basedir 'Functions/taup/lib/TauP-1.1.7.jar']);
javaaddpath([basedir 'Functions/taup/lib/log4j-1.2.8.jar']);
javaaddpath([basedir 'Functions/taup/lib/seisFile-1.0.1.jar']);
addpath([basedir 'Functions/Matlab_TauP']);
addpath([basedir 'Functions/SACfun']); addpath([basedir 'Functions']);
cd(basedir)


latlon = [-10.5 34.5];
rf_phases = {'Ps','Sp'};

Project = 'SEGMeNT_4';
datadir=[basedir 'Data\CCP\' Project '\CCP\']; 
Ps=load([datadir 'meanbtstrapPs22.mat']);
Sp=load([datadir 'meanbtstrap97.mat']); 

phase_vel_name = 'SEGMeNT_phv.txt';



% Output RF for RJMCMC

% Identify nearest station and the migration model used
Projdir=[basedir 'Data/Projects/' Project '/'];
load([Projdir 'Networks.mat']);
slocs = zeros(150,2); snames{150,1} = ''; 
nets = fieldnames(Networks); n=0;
for in = 1:length(nets); stas = fieldnames(Networks.(nets{in}));
    for iis = 1:length(stas); n=n+1; snames{n} = [nets{in} '.' stas{iis}];
        slocs(n,:)=[Networks.(nets{in}).(stas{iis}).Latitude, ...
            Networks.(nets{in}).(stas{iis}).Longitude];
    end
end
slocs = slocs(1:n,:); snames=snames(1:n);
[d, ind] = min(distance(slocs(:,1), slocs(:,2), latlon(1), latlon(2)));
if d > 1; disp('Uh oh!  Chosen point really far from any stations!'); end
staname = strsplit(snames{ind},'.');
load([Projdir staname{1} '/' staname{2} '/Migration_Models.mat']);
titstr = [num2str(abs(latlon(1))) '\circS, ' num2str(latlon(2)) '\circE', ...
    ' (' snames{ind} ')'];

% Retrieve RFs
for irf = 1:length(rf_phases)
    rf_phase = rf_phases{irf};
    switch rf_phase
        case 'Ps'; RF = Ps; ind_rad = 0.15;
        case 'Sp'; RF = Sp; ind_rad = 0.25;
    end
    
    % Find lat/lon points that are within 0.2 degrees of desired point
    inds = find(abs(RF.model_lons - latlon(2))<ind_rad & ...
        abs(RF.model_lats - latlon(1))<ind_rad);
    
    rfs = RF.plane_RF(inds,:); rfs(rfs==0) = nan;
    
    time = interp1(Migration_Models.(rf_phase).Reference_Depth,...
        Migration_Models.(rf_phase).Reference_DT, RF.cp_depths);
    rf_t = [time', nanmedian(rfs,1)'];
    std = RF.std_RF(inds,:); std(std==0) = nan;
    std = nanmedian(std,1)';
    
    t_fine = linspace(time(1),time(end),1e3);
    rf_fine = interp1(rf_t(:,1),rf_t(:,2),t_fine);
    std_fine = interp1(rf_t(:,1),std,t_fine);
    
    switch rf_phase
        case 'Ps'
            plot_RF_with_std(time, rfs, t_fine, rf_fine, ...
                std_fine, max(RF.cp_depths), titstr);    
        case 'Sp'
            plot_RF_with_std(-time, -rfs, -t_fine, -rf_fine, ...
                std_fine, max(RF.cp_depths), titstr); 
    end
    
    fldname = ['rf' num2str(irf)];
    
    switch rf_phase
        case 'Ps'
            allrfs.(fldname).rf = rf_t; allrfs.(fldname).std = std; 
            allrfs.(fldname).inds = inds; allrfs.(fldname).flip = 0;
            allrfs.(fldname).filter_corners = [1,100]; 
            allrfs.(fldname).rp = Migration_Models.(rf_phase).Reference_RP;
        case 'Sp'
            allrfs.(fldname).rf = rf_t; allrfs.(fldname).std = std; 
            allrfs.(fldname).inds = inds; allrfs.(fldname).flip = 1;
            allrfs.(fldname).filter_corners = [4,100]; 
            allrfs.(fldname).rp = Migration_Models.(rf_phase).Reference_RP;
    end
end


% Plot map showing area averaged over
figure; hold on; box on
lon=[33.5 36.1]; lat=[-11.6 -8.5];
Countries=shaperead([basedir 'Data/Misc/ne_50m_admin_0_countries.shp'],'UseGeoCoords',true);
Lakes=shaperead([basedir 'Data/Misc/ne_50m_lakes.shp'],'UseGeoCoords',true);
for ic = 1:length(Countries)
    plot(Countries(ic).Lon,Countries(ic).Lat,'k-'); 
end
for il = [6,10,164]%1:length(Lakes)
    jinds=[1 find(isnan(Lakes(il).Lon) | isnan(Lakes(il).Lat))...
        length(Lakes(il).Lon)];
    for k=1:length(jinds)-1
        inds=jinds(k)+1:jinds(k+1)-1;
        fill(Lakes(il).Lon(inds),Lakes(il).Lat(inds),'w');
        if k==1
            h=fill(Lakes(il).Lon(inds),Lakes(il).Lat(inds),'b'); 
            set(h,'facealpha',0.5);
        end
    end
    plot(Lakes(il).Lon,Lakes(il).Lat,'k-');
end
daspect([111.16, 111.16*distance(mean(lat),0,mean(lat),1), 1]);
axis([lon lat]);

plot(Sp.model_lons(allrfs.rf2.inds),Sp.model_lats(allrfs.rf2.inds),'k.');
plot(Ps.model_lons(allrfs.rf1.inds),Ps.model_lats(allrfs.rf1.inds),'r.');


%% Load in the phase velocities
filename=[basedir 'Data/Velocity_Models/PhaseVels/SEGMeNT_phv.txt'];
delimiter={' ','\t'};formatSpec = '%s%s%s%s%s%[^\n\r]'; fileID=fopen(filename,'r');
dataArray=textscan(fileID, formatSpec,'Delimiter', delimiter,...
    'MultipleDelimsAsOne',true,'HeaderLines', 1, 'ReturnOnError', false);
fclose(fileID); phv = [dataArray{1:end-1}]; phv=cellfun(@(x)str2double(x),phv);
periods=unique(phv(:,3));

c = zeros(length(periods),1); c_std = c;
for iipv = length(periods):-1:1
    inds=find(phv(:,3)==periods(iipv));
    ind=find(abs(phv(inds,1)-latlon(1))<0.1 & abs(phv(inds,2)-latlon(2))<0.1);
    if isempty(ind)
        ind=find(abs(phv(inds,1)-latlon(1))<0.2 & ...
            abs(phv(inds,2)-latlon(2))<0.2); ind=ind(1); 
    end
    c(iipv) = mean(phv(inds(ind),4));
    c_std(iipv) = mean(phv(inds(ind),5));
end


%% Print for copying into input_data.py

dt = 0.25; std_sc = 5;



clc; close all; ifplot = 0;

fprintf('\n\nimport pipeline\nimport numpy as np\n\n');

fprintf('\ndef LoadObservations():\n\n');
fprintf('\t#   %.1f\x00b0 S, %.1f\x00b0 E\n\n', latlon(1),latlon(2));
fprintf('\trf_obs = [')
for irf = 1:length(rf_phases)
    
    rfnames = fieldnames(allrfs); rf_out = allrfs.(rfnames{irf});
    if rf_out.flip; coarse_time = -dt:-dt:-30; else; coarse_time = 0:dt:30-dt; end
    rf = interp1(rf_out.rf(:,1), rf_out.rf(:,2), coarse_time); 
    std = interp1(rf_out.rf(:,1), rf_out.std, coarse_time);
    std(isnan(rf)) = 1; rf(isnan(rf)) = 0;
    
    fprintf('\n\t\tpipeline.RecvFunc(\n\t\t\t\tamp = np.array([');
    printforpython(rf,3); fprintf('\t\t\t\t\t]),');
    fprintf('\n\t\t\t\tstd = np.array([');
    printforpython(std,3); fprintf('\t\t\t\t\t]),');
    fprintf('\n\t\t\t\tdt = %.3f, ray_param = %.5f,', dt, rf_out.rp);
    fprintf('\n\t\t\t\tstd_sc = %g, rf_phase = ''%s'',', std_sc, rf_phases{irf})
    fprintf('\n\t\t\t\tfilter_corners = [%g, %g]\n\t\t\t\t),\n', ...
        rf_out.filter_corners(1), rf_out.filter_corners(2))
    
    if ifplot
        figure; hold on; plot(rf,coarse_time,'k-','linewidth',2)
        plot(rf-std,coarse_time,'--','color',0.75*[1 1 1])
        plot(rf+std,coarse_time,'--','color',0.75*[1 1 1])
        plot([0 0],coarse_time([1 end]),':','color',0.2*[1 1 1]);
        if ~rf_out.flip; axis ij; else; set(gca,'xdir','reverse'); end
        xlim(0.3*[-1 1]);
        title(rf_phases{irf}); xlabel('RF Amplitude'); ylabel('Time (s)');
    end
end
fprintf('\t\t]\n\n\n');

fprintf('\tswd_obs = pipeline.SurfaceWaveDisp(\n\t\t\t\tperiod = np.array([');
printforpython(periods,3)
fprintf('\t\t\t\t\t]),\n\t\t\t\tc = np.array([')
printforpython(c,4)
fprintf('\t\t\t\t\t])\n\t\t\t\t)\n\n');
if ifplot
    figure; plot(periods,c); 
    xlabel('Period (s)'); ylabel('Phase Velocity (km/s)');
end

fprintf('\tall_lims = pipeline.Limits(\n\t\t\t\tvs = (0.5, %.1f),', ...
    min(5,0.55/rf_out.rp)) % ensures Vp < c always
fprintf('dep = (0,200), std_rf = (0, 0.05),\n\t\t\t\tlam_rf = (0.05')
fprintf(', 0.5), std_swd = (0, 0.05), crustal_thick = (25,))');

fprintf('\n\n\tvs_in = ''unknown''')

fprintf('\n\n\treturn (rf_obs, swd_obs, all_lims, vs_in)\n\n\n');













%% Plot phase velocities

Countries=shaperead([basedir 'Data/Misc/ne_50m_admin_0_countries.shp'],'UseGeoCoords',true);
Lakes=shaperead([basedir 'Data/Misc/ne_50m_lakes.shp'],'UseGeoCoords',true);


figure('color','w','position',[1,41,1280,607]);
for ip = 1:length(periods) % 12
    
    subplot(3,4,ip); hold on
    
    inds = find(phv(:,3)==periods(ip) & phv(:,2)>33);
    scatter(phv(inds,2),phv(inds,1),10,phv(inds,5),'filled','o');
    c= colorbar('location','eastoutside'); ylabel(c,'Std');%ylabel(c,'Phase Velocity (km/s)');
    
    for ic = 1:length(Countries); plot(Countries(ic).Lon,Countries(ic).Lat,'k-'); end
    for il = 1:length(Lakes); plot(Lakes(il).Lon,Lakes(il).Lat,'b-'); end
    box on; set(gca, 'xgrid','on','ygrid','on','layer','top');
    axis([33, 35,-12, -8.5 ])
    daspect([1/distance(-11,33,-11,34) 1/distance(-11,33,-12,33) 1])
    %xlabel('Longitude (\degE)'); ylabel('Latitude (\deg N)');
    title([num2str(periods(ip)) ' s']); caxis([0 0.05]);
    
end
colormap(flipud(jet))



