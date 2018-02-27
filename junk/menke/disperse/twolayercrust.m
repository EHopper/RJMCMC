% Input file for mat_disperse.m

D = [ 0.02	4.0969666667	4.1940173913;
      0.03	3.8713333333	3.9399;
      0.04	3.8999333333	3.7413086957;
      0.05	3.6883666667	3.6243217391;
      0.06	3.6320888889	3.3558043478;
      0.07	3.5967555556	3.0461304348;
      0.08	3.5293111111	2.9623565217;
      0.09	3.5470111111	2.7868217391;
      0.1	3.4109		2.575373913 ];

% Define a vector of frequencies (in Hz)
freq = D(:,1)';
Nf = length(freq);

% Call mat_disperse.m to solve the eigenvalue problem and calculate phase
% velocities, displacement-stress functions, and surface wave displacements
vrref = mat_disperse(thk,dns,vp,vs,freq);

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
xlabel('VpTopCrust');
vpc = ginput(1);
plot( [vpc(1), vpc(1)], [0, 1], 'k-', 'LineWidth', 2 );

figure(4);
clf;
set(gca,'LineWidth',2);
hold on;
axis( [0, 10, 0 1]  );
xlabel('VpBotCrust');
vpc2 = ginput(1);
plot( [vpc2(1), vpc2(1)], [0, 1], 'k-', 'LineWidth', 2 );

figure(5);
clf;
set(gca,'LineWidth',2);
hold on;
axis( [0, 10, 0 1]  );
xlabel('VpMantle');
vpm = ginput(1);
plot( [vpm(1), vpm(1)], [0, 1], 'k-', 'LineWidth', 2 );

% another earth model
thk = [t(1)/2 t(1)/2];
dns = [2.5 2.75 3.0];
r = 1.78;
vp = [vpc(1) vpc2(1) vpm(1)];
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

fprintf('T=%.1f Vct=%.2f Vcb=%.2f Vm=%.2f E1=%.3f E2=%.3f\n', thk(1), vp(1), vp(2), vp(3), E1, E2 );

x = ginput(1);
if( x(1) < 0.01 )
    break;
end

end

