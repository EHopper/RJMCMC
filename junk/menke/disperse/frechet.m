% Input file for mat_disperse.m
clear all;

% earth model
thk = [10 10 10];
dns = [2.5 2.5 2.5 3.0];
r = 1.78;
vp = [6.0 6.0 6.0 8.0];
vs = vp/r;

% Define a vector of frequencies (in Hz)
freq = [0.01:0.01:0.10]';
fmax=max(freq);
Nf = length(freq);

% Call mat_disperse.m to solve the eigenvalue problem and calculate phase
% velocities, displacement-stress functions, and surface wave displacements
vr0 = mat_disperse(thk,dns,vp,vs,freq);
Ds=0.01;
L=1;
vs(L)=vs(L)+ Ds;
vr1 = mat_disperse(thk,dns,vp,vs,freq);
vs(L)=vs(L)- Ds;

figure(1);
clf;
set(gca,'LineWidth',2);
hold on;
axis( [0, fmax, 0, 5]  );
xlabel('f, Hz');
ylabel('v, km/s');
plot( freq, vr0(:,1), 'k-', 'LineWidth', 2 );
plot( freq, vr1(:,1), 'r-', 'LineWidth', 2 );
dvdvs=(vr1(:,1)-vr0(:,1))/Ds;

figure(2);
clf;
set(gca,'LineWidth',2);
hold on;
axis( [0, fmax, 0, max(dvdvs)]  );
xlabel('f, Hz');
ylabel('dv/ds, (km/s)/vs');
plot( freq, dvdvs, 'k-', 'LineWidth', 2 );


