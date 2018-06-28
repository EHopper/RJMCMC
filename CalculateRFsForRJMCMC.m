clearvars; close all; clc
basedir='C:\Users\emily\OneDrive\Documents\WORK\MATLAB\Scattered_Waves\';
javaaddpath([basedir 'Functions/taup/lib/TauP-1.1.7.jar']);
javaaddpath([basedir 'Functions/taup/lib/log4j-1.2.8.jar']);
javaaddpath([basedir 'Functions/taup/lib/seisFile-1.0.1.jar']);
addpath([basedir 'Functions/Matlab_TauP']);
addpath([basedir 'Functions/SACfun']); addpath([basedir 'Functions']);
cd(basedir)


Project = 'SEGMeNT_5';
staname = 'YQ.MBEY';
rf_phases = {'Ps','Sp'};

phase_vel_name = 'SEGMeNT_phv.txt';


%% Output RF for RJMCMC
Projdir=[basedir 'Data/Projects/' Project '/'];
load([Projdir 'Networks.mat']);

staname_split = strsplit(staname, '.');
net = staname_split{1}; sta = staname_split{2};
stadir = [Projdir net '/' sta '/'];

% Prep the waveform
Auto_Prep(Project,basedir,'prep',{net}, 'Best', [], [], {sta});
clc
%%
for k = 1:length(rf_phases)
    rf_phase = rf_phases{k};
    switch rf_phase
        case 'Sp'; filter_corners = [4,100];
        case 'Ps'; filter_corners = [1,100];
    end
    
    %% Calculate the RF
    CrustModel = 'None';
    MantleVpModel = 'Accardo2017Expanded';
    MantleVsModel = 'Accardo2017Expanded';
    SvSh = 'SV';
    ifRZ = 0;
    filt = 'bp';
    Cull = [];
    version = 'namesForRJMCMC';
    Generate_RFs(Project, basedir, {staname}, {rf_phase}, 'generate', ...
        filter_corners,CrustModel, MantleVpModel, MantleVsModel, ...
        SvSh, ifRZ, Cull, filt, 'MTM', version);
    clc
    
    %% Load in and remove outliers
    load([Projdir version '.mat'])
    load([stadir 'PreCalculated_RFs' FileNames.PreCalcRFs]);
    
    RF = migrate_single_station_stack_from_PreCalc(...
        PreCalculated_RFs, stadir, rf_phase);
    
    cull=8;
    plot_single_sta_RF(RF.RFs, RF.TIME, RF.DEPTH, cull, rf_phase);
    
    %%
    good = 'y';
    while strcmp(good,'y')
        nk = 2;
        rfs = RF.RFs; baz = RF.BAZ; dist = RF.GCARC; evs = RF.EVS; rp = RF.RP;
        switch rf_phase
            case 'Sp'; time = -RF.TIME; maxamp = 0.3;
            case 'Ps'; time = RF.TIME; maxamp = 0.6;
        end
        for id=size(rfs,1):-1:1
            if sum(abs(rfs(id,:))>maxamp); rfs(id,:)=[]; end
        end
        
        for iik=1:10
            [icl, ccl] = kmeans(rfs,nk);
            figure('position',[30,30,1150,600],'color','w')
            [t,~] = meshgrid(time, ones(size(rfs(:,1))));
            axes('position',[0.1 0.3 0.25 0.65]);
            plot(rfs',t','b-'); hold on; set(gca,'ydir','reverse');
            ylim([0, 30]);  plot(ccl',t(1:nk,:)','k-','linewidth',2)
            plot([0,0],[0,30],'--','color',0.5*[1 1 1])
            xlabel('RF Amplitude'); ylabel('Time (s)')
            axes('position',[0.1 0.1 0.1 0.1]);
            hist(dist); xlabel('Distance'); ylabel('Frequency'); xlim([30 90])
            axes('position',[0.21 0.1 0.1 0.1]);
            polarhistogram(deg2rad(baz),24,'facecolor','b','facealpha',0.6)
            ax=gca; ax.ThetaDir='clockwise'; ax.ThetaZeroLocation='top';
            ax.ThetaTick = 0:90:360;
            
            axes('position',[0.4,0.3,0.25,0.65]);
            plot(rfs(icl==1,:)',t(icl==1,:)','b-'); hold on;
            set(gca,'ydir','reverse'); ylim([0 30])
            plot(ccl(1,:)',t(1,:)','k-','linewidth',2)
            title(['ONE: ' num2str(sum(icl==1)) ' rfs']);
            plot([0,0],[0,30],'--','color',0.5*[1 1 1])
            xlabel('RF Amplitude'); ylabel('Time (s)')
            axes('position',[0.4 0.1 0.1 0.1]);
            hist(dist(icl==1)); xlabel('Distance'); ylabel('Frequency'); xlim([30 90])
            axes('position',[0.51 0.1 0.1 0.1]);
            polarhistogram(deg2rad(baz(icl==1)),24,'facecolor','b','facealpha',0.6)
            ax=gca; ax.ThetaDir='clockwise'; ax.ThetaZeroLocation='top';
            ax.ThetaTick = 0:90:360;
            
            axes('position',[0.7,0.3,0.25,0.65]);
            plot(rfs(icl==2,:)',t(icl==2,:)','b-'); hold on;
            set(gca,'ydir','reverse'); ylim([0 30])
            plot(ccl(2,:)',t(1,:)','k-','linewidth',2)
            title(['TWO: ' num2str(sum(icl==2)) ' rfs']);
            plot([0,0],[0,30],'--','color',0.5*[1 1 1])
            xlabel('RF Amplitude'); ylabel('Time (s)')
            axes('position',[0.7 0.1 0.1 0.1]);
            hist(dist(icl==2)); xlabel('Distance'); ylabel('Frequency'); xlim([30 90])
            axes('position',[0.81 0.1 0.1 0.1]);
            polarhistogram(deg2rad(baz(icl==2)),24,'facecolor','b','facealpha',0.6)
            ax=gca; ax.ThetaDir='clockwise'; ax.ThetaZeroLocation='top';
            ax.ThetaTick = 0:90:360;
            
            
            which_c = input('Which cluster is better? 1 or 2?  (Press 3 to stop). ');
            if which_c == 3; close; break; end
            rfs = rfs(icl==which_c,:); t = t(icl==which_c,:);
            baz = baz(icl == which_c); dist = dist(icl==which_c);
            evs = evs(icl == which_c);
            close
        end
        
        
        
        rf_out = plot_single_sta_RF(rfs, RF.TIME, RF.DEPTH, cull, ...
            rf_phase, filter_corners);
        figure('position',[980 90 550 700],'color','w');
        subplot(2,1,1); hist(dist); xlabel('Distance'); title('Distance')
        subplot(2,1,2); polarhistogram(deg2rad(baz),24,'facecolor','b','facealpha',0.6)
        ax=gca; ax.ThetaDir='clockwise'; ax.ThetaZeroLocation='top';
        ax.ThetaTick = 0:45:360; title('Backazimuth')
        
        good = input('Redo this? y/[n] ','s'); close
    end
    
    
    switch rf_phase
        case 'Ps'
            ps.rf = rf_out; ps.baz = baz; ps.evs = evs; ps.dist = dist;
            ps.filter_corners = filter_corners; ps.rp = RF.RP;
        case 'Sp'
            sp.rf = rf_out; sp.baz = baz; sp.evs = evs; sp.dist = dist;
            sp.filter_corners = filter_corners; sp.rp = rp;
    end
end

%% Load in the phase velocities
filename=[basedir 'Data/Velocity_Models/PhaseVels/SEGMeNT_phv.txt'];
delimiter={' ','\t'};formatSpec = '%s%s%s%s%s%[^\n\r]'; fileID=fopen(filename,'r');
dataArray=textscan(fileID, formatSpec,'Delimiter', delimiter,...
    'MultipleDelimsAsOne',true,'HeaderLines', 1, 'ReturnOnError', false);
fclose(fileID); phv = [dataArray{1:end-1}]; phv=cellfun(@(x)str2double(x),phv);
periods=unique(phv(:,3));

SLOC = [Networks.(net).(sta).Latitude, Networks.(net).(sta).Longitude];
c = zeros(length(periods),1); c_std = c;
for iipv = length(periods):-1:1
    inds=find(phv(:,3)==periods(iipv));
    ind=find(abs(phv(inds,1)-SLOC(1))<0.1 & abs(phv(inds,2)-SLOC(2))<0.1);
    if isempty(ind);ind=find(abs(phv(inds,1)-SLOC(1))<0.2 & abs(phv(inds,2)-SLOC(2))<0.2); ind=ind(1); end
    c(iipv) = mean(phv(inds(ind),4));
    c_std(iipv) = mean(phv(inds(ind),5));
end


%% Print for copying into input_data.py

dt = 0.25; std_sc = 5;


if length(rf_phases) == 1
    
    
    rf = interp1(rf_out(1,:), rf_out(2,:), 0:dt:30);
    
    clc
    fprintf('\n\nimport pipeline\nimport numpy as np\n\n');
    
    fprintf('\ndef LoadObservations():\n\n');
    fprintf('\trf_obs = pipeline.RecvFunc(\n\t\t\t\tamp = np.array([');
    if strcmp(rf_phase,'Ps'); printforpython(rf(1:end-1),3)
    elseif strcmp(rf_phase,'Sp'); printforpython(rf(2:end),3)
    end
    fprintf('\t\t\t\t\t]),\n\t\t\t\tdt = %.3f, ray_param = %.5f,', dt, RF.RP);
    fprintf('\n\t\t\t\tstd_sc = %g, rf_phase = ''%s'',', std_sc, rf_phase)
    fprintf('\n\t\t\t\tfilter_corners = [%g, %g]\n\t\t\t\t)\n\n', filter_corners(1), filter_corners(2))
    
    fprintf('\tswd_obs = pipeline.SurfaceWaveDisp(\n\t\t\t\tperiod = np.array([');
    printforpython(periods,3)
    fprintf('\t\t\t\t\t]),\n\t\t\t\tc = np.array([')
    printforpython(c,4)
    fprintf('\t\t\t\t\t])\n\t\t\t\t)\n\n');
    
    fprintf('\tall_lims = pipeline.Limits(\n\t\t\t\tvs = (0.5, %.1f),', min(5,0.55/RF.RP)) % ensures Vp < c always
    fprintf('dep = (0,200), std_rf = (0, 0.05),\n\t\t\t\tlam_rf = (0.05')
    fprintf(', 0.5), std_swd = (0, 0.05), crustal_thick = (25,))');
    
    fprintf('\n\n\treturn (rf_obs, swd_obs, all_lims)\n\n\n');
    
    for ie = 1:length(evs)
        fprintf('#  %s\t\tBAZ: %.0f%c\t\tDIST: %.0f%c\n',evs{ie},baz(ie),char(176),...
            dist(ie), char(176))
    end
    
    
else
    
    clc
    fprintf('\n\nimport pipeline_three\nimport numpy as np\n\n');

    fprintf('\ndef LoadObservations():\n\n');
    
    fprintf('\tswd_obs = pipeline_three.SurfaceWaveDisp(\n\t\t\t\tperiod = np.array([');
    printforpython(periods,3)
    fprintf('\t\t\t\t\t]),\n\t\t\t\tc = np.array([')
    printforpython(c,4)
    fprintf('\t\t\t\t\t])\n\t\t\t\t)\n\n');
    
    fprintf('\tall_lims = pipeline_three.Limits3(\n\t\t\t\tvs = (0.5, %.1f),', ...
        min(5,0.55/RF.RP)) % ensures Vp < c always
    fprintf('dep = (0,200), std_rf_Ps = (0, 0.05),\n\t\t\t\tlam_rf_Ps = (0.05,0.5),')
    fprintf('std_rf_Sp = (0, 0.05),lam_rf_Sp = (0.05, 0.5),')
    fprintf('\n\t\t\t\tstd_swd = (0, 0.05), crustal_thick = (25,))');

    
    for k = 1:length(rf_phases)
        rf_phase = rf_phases{k};
        switch rf_phase
            case 'Ps'
                rf_out = ps.rf; evs = ps.evs; dist = ps.dist; rp = ps.rp; 
                baz = ps.baz; filter_corners = ps.filter_corners;
            case 'Sp'
                rf_out = sp.rf; evs = sp.evs; dist = sp.dist; rp = sp.rp;
                baz = sp.baz; filter_corners = sp.filter_corners;
        end
        rf = interp1(rf_out(1,:), rf_out(2,:), 0:dt:30);
    
        if strcmp(rf_phase,'Ps')
            fprintf('\n\n\trf_obs_Ps = pipeline_three.RecvFunc(\n\t\t\t\tamp = np.array([');
            printforpython(rf(1:end-1),3)
        elseif strcmp(rf_phase,'Sp')
            fprintf('\n\n\trf_obs_Sp = pipeline_three.RecvFunc(\n\t\t\t\tamp = np.array([');
            printforpython(rf(2:end),3)
        end
        fprintf('\t\t\t\t\t]),\n\t\t\t\tdt = %.3f, ray_param = %.5f,', dt, rp);
        fprintf('\n\t\t\t\tstd_sc = %g, rf_phase = ''%s'',', std_sc, rf_phase)
        fprintf('\n\t\t\t\tfilter_corners = [%g, %g]\n\t\t\t\t)\n\n',...
            filter_corners(1), filter_corners(2))
        
        for ie = 1:length(evs)
            fprintf('\t\t\t\t#  %s\t\tBAZ: %.0f%c\t\tDIST: %.0f%c\n',...
                evs{ie},baz(ie),char(176),dist(ie), char(176))
        end
        
    end
    
    fprintf('\n\n\treturn (rf_obs_Ps, rf_obs_Sp, swd_obs, all_lims)\n\n\n');
    

end










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



