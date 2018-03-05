% The array D has two hypothetical data urves in it dispersion curves it.
% In the example, one clicks on three plots to pick a layer-over-halfspace
% model (H-crust, Vp-crust and Vp-mantle, with Vs scaled by a factor of 1.78)
% to try to fit either of the data curves "by hand".  Bill Menke, 2013.

D = [ 0.02	4.0969666667	4.1940173913;
      0.03	3.8713333333	3.9399;
      0.04	3.8999333333	3.7413086957;
      0.05	3.6883666667	3.6243217391;
      0.06	3.6320888889	3.3558043478;
      0.07	3.5967555556	3.0461304348;
      0.08	3.5293111111	2.9623565217;
      0.09	3.5470111111	2.7868217391;
      0.1	3.4109		2.575373913 ];
  

%% earth model
clearvars;
tic
thk = '';
% dns = [2.5 3.0];
% r = 1.78;
% vp = [6.0 8.0];
% vs = vp/r;
vs = [4., 1.2, 4.3, 2.7, 3.1]; idep = [2 17 42 58 91]+1;

addpath('C:\Users\Emily\OneDrive\Documents\WORK\MATLAB\Scattered_Waves\RJMCMC\junk\');
[vp, dns, thk] = makevelmodel(vs, idep); thk=thk(1:end-1);

% Define a vector of frequencies (in Hz)
freq = 0.02:0.01:0.1;%D(:,1)';
Nf = length(freq);
fmax = max(freq);

% Call mat_disperse.m to solve the eigenvalue problem and calculate phase
% velocities, displacement-stress functions, and surface wave displacements
vrref = mat_disperse(thk,dns,vp,vs,freq);
fprintf('\n\n')
for k = 1:ceil(length(freq)/3)
   fprintf('%.4f, %.4f, %.4f,\n',vrref(3*k-2,1), vrref(3*k-1,1),...
       vrref(3*k,1));
    
end
toc
%%
for itt = [1:100]

figure(2)
clf;
set(gca,'LineWidth',2);
hold on;
axis( [0, 50, 0 1]  );
xlabel('thickness');
t = ginput(1);
plot( [t(1), t(1)], [0, 1], 'k-', 'LineWidth', 2 );

figure(3);
clf;
set(gca,'LineWidth',2);
hold on;
axis( [0, 10, 0 1]  );
xlabel('VpCrust');
vpc = ginput(1);
plot( [vpc(1), vpc(1)], [0, 1], 'k-', 'LineWidth', 2 );

figure(4);
clf;
set(gca,'LineWidth',2);
hold on;
axis( [0, 10, 0 1]  );
xlabel('VpMantle');
vpm = ginput(1);
plot( [vpm(1), vpm(1)], [0, 1], 'k-', 'LineWidth', 2 );

% another earth model
thk = [t(1)];
dns = [2.5 3.0];
r = 1.78;
vp = [vpc(1) vpm(1)];
vs = vp/r;

% Call mat_disperse.m to solve the eigenvalue problem and calculate phase
% velocities, displacement-stress functions, and surface wave displacements
vr = mat_disperse(thk,dns,vp,vs,freq);

figure(1);
clf;
set(gca,'LineWidth',2);
hold on;
axis( [0, fmax, 0 5]  );
xlabel('f, Hz');
ylabel('v, km/s');
plot( freq, D(:,2)', 'ro', 'LineWidth', 2 );
plot( freq, D(:,3)', 'bo', 'LineWidth', 2 );
plot( freq, vr(:,1), 'k-', 'LineWidth', 2 );

e1 = (vr(:,1)-D(:,2));
E1 = sqrt(e1'*e1/Nf);
e2 = (vr(:,1)-D(:,3));
E2 = sqrt(e2'*e2/Nf);

fprintf('T=%.1f Vc=%.2f Vm=%.2f E1=%.3f E2=%.3f\n', thk(1), vp(1), vp(2), E1, E2 );

x = ginput(1);
if( x(1) < 0.01 )
    break;
end

end

