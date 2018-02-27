clear all;

freq = [ 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 ];

% earth model
Dh=0.0;
thk = [3.0+Dh, 8.0];
dns = [1.0, 2.5, 3.0];
vp = [1.5, 6.0, 8.0];
r = [1.78, 1.78, 1.78];
vs = vp./r;

% Define a vector of frequencies (in Hz)
Nf = length(freq);
fmax = max(freq);

% Call mat_disperse.m to solve the eigenvalue problem and calculate phase
% velocities, displacement-stress functions, and surface wave displacements
vr1 = mat_disperse(thk,dns,vp,vs,freq);

% earth model
Dh=0.1;
thk = [3.0+Dh, 8.0];
vr2 = mat_disperse(thk,dns,vp,vs,freq);

figure(1);
clf;
set(gca,'LineWidth',2);
hold on;
axis( [0, fmax, 0 5]  );
xlabel('f, Hz');
ylabel('v, km/s');
plot( freq, vr1(:,1), 'k-', 'LineWidth', 2 );
plot( freq, vr2(:,1), 'k-', 'LineWidth', 2 );

dvov = (vr1(:,1)-vr2(:,1))./((vr1(:,1)+vr2(:,1))/2);

figure(2);
clf;
set(gca,'LineWidth',2);
hold on;
axis( [0, fmax, -0.1 0.1]  );
xlabel('f, Hz');
ylabel('v, km/s');
plot( freq, dvov, 'k-', 'LineWidth', 2 );


