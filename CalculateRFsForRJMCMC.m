clearvars; close all; clc
basedir='C:\Users\emily\OneDrive\Documents\WORK\MATLAB\Scattered_Waves\';
addpath([basedir 'Functions\']);


Project = 'SEGMeNT_4';
staname = 'AF.MBEY';
rf_phase = 'Sp';
filter_corners = [1,100];
phase_vel_name = 'SEGMeNT_phv.txt';

%%
for k = 1
%% Output RF for RJMCMC
Projdir=[basedir 'Data/Projects/' Project '/'];
load([Projdir 'Networks.mat']);

staname_split = strsplit(staname, '.');
net = staname_split{1}; sta = staname_split{2};
stadir = [Projdir net '/' sta '/'];

% Prep the waveform
Auto_Prep(Project,basedir,'prep',{net}, 'Best', [], [], {sta});
clc


% Calculate the RF
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

cull=10;
rf_out = plot_single_sta_RF(RF.RFs, RF.TIME, RF.DEPTH, cull, ...
    rf_phase, filter_corners);

%% Load in the phase velocities
filename=[basedir 'Data/Velocity_Models/PhaseVels/SEGMeNT_phv.txt'];
delimiter={' ','\t'};formatSpec = '%s%s%s%s%[^\n\r]'; fileID=fopen(filename,'r');
dataArray=textscan(fileID, formatSpec,'Delimiter', delimiter,...
    'MultipleDelimsAsOne',true,'HeaderLines', 1, 'ReturnOnError', false);
fclose(fileID); phv = [dataArray{1:end-1}]; phv=cellfun(@(x)str2double(x),phv);
periods=unique(phv(:,3)); 

SLOC = [Networks.(net).(sta).Latitude, Networks.(net).(sta).Longitude];
c = zeros(length(periods),1); 
for iipv = length(periods):-1:1
    inds=find(phv(:,3)==periods(iipv));
    ind=find(abs(phv(inds,1)-SLOC(1))<0.1 & abs(phv(inds,2)-SLOC(2))<0.1);
    if isempty(ind);ind=find(abs(phv(inds,1)-SLOC(1))<0.2 & abs(phv(inds,2)-SLOC(2))<0.2); ind=ind(1); end
    c(iipv) = phv(inds(ind),4);
end
    

%% Print for copying into input_data.py

dt = 0.25; std_sc = 5;
rf = interp1(rf_out(1,:), rf_out(2,:), 0:dt:30);

clc
fprintf('\ndef LoadObservations():\n\n');
fprintf('\trf_obs = pipeline.RecvFunc(\n\t\t\t\tamp = np.array([');
if rf_phase == 'Ps'; printforpython(rf(1:end-1),3)
elseif rf_phase == 'Sp'; printforpython(rf(2:end),3)
end
fprintf('\t\t\t\t\t]),\n\t\t\t\tdt = %.3f, ray_param = %.5f,', dt, RF.RP);
fprintf('\n\t\t\t\tstd_sc = %g, rf_phase = ''%s'',', std_sc, rf_phase)
fprintf('\n\t\t\t\tfilter_corners = [%g, %g]\n\t\t\t\t)\n\n', filter_corners(1), filter_corners(2))

fprintf('\tswd_obs = pipeline.SurfaceWaveDisp(\n\t\t\t\tperiod = np.array([');
printforpython(periods,3)
fprintf('\t\t\t\t\t]),\n\t\t\t\tc = np.array([')
printforpython(c,4)
fprintf('\t\t\t\t\t])\n\t\t\t\t)\n\n');

fprintf('\tall_lims = pipeline.Limits(\n\t\t\t\tvs = (0.5, 5.5),')
fprintf('dep = (0,200), std_rf = (0, 0.05),\n\t\t\t\tlam_rf = (0.05')
fprintf(', 0.5), std_swd = (0, 0.15))');

fprintf('\n\n\treturn (rf_obs, swd_obs, all_lims)\n\n');
% ]), dt = 0.25, ray_param = 0.06147,
%                         std_sc = 5, rf_phase = 'Ps', filter_corners = [1,100])
% 
%     swd_obs = pipeline.SurfaceWaveDisp(period = np.array([9.0, 10.1, 11.6, 13.5,
%                         16.2, 20.3, 25.0, 32.0, 40.0, 50.0, 60.0, 80.0]),
%                         c = np.array([3.212, 3.215, 3.233, 3.288,
%                        3.339, 3.388, 3.514, 3.647, 3.715, 3.798, 3.847, 3.937]))
% 
%     all_lims = pipeline.Limits(
%         vs = (0.5,5.5), dep = (0,200), std_rf = (0,0.05),
%         lam_rf = (0.05, 0.5), std_swd = (0,0.15))
% 
%     return (rf_obs, swd_obs, all_lims)

end