function [vp,rho,t]=makevelmodel(vs,idep)

%clearvars; close all; clc
%vs=[4.5, 4.6, 4.7, 5.];
deps=[0:0.2:10, 11:60, 65:5:200];
%idep=[60,80,108,120]+1;

dep=deps(idep);
vp=zeros(size(vs)); rho=vp; rsc=vp; dr=vp; t=vp;
t(1:end-1)=(dep(1:end-1)-[-dep(1) dep(1:end-2)])/2+...
    (dep(2:end)-dep(1:end-1))/2;
if length(vs == 1); avdep = t; 
else; avdep=cumsum([0 t(1:end-1)])+t([1:end-1 end-1])/2;
end

ic=find(vs<4.5); 
if isempty(ic); mi =1; ice=0; else; ice=ic(end); mi = ice+1; end
iumm=find(avdep(mi:end)<135)+ice;
ium=find(avdep(mi:end)>135)+ice;


vp(mi:end)=vs(mi:end).*1.75;
vp(ic)=0.9409+2.0947.*vs(ic) - 0.8206.*vs(ic).^2 +...
    0.2683.*vs(ic).^3 -0.0251.*vs(ic).^4;

rho(ic)=1.6612.*vp(ic)-0.4721.*vp(ic).^2+0.0671.*vp(ic).^3-...
    0.0043.*vp(ic).^4+0.000106.*vp(ic).^5;
rsc(iumm)=2.4e-4*avdep(iumm)+5.6e-2;
rsc(ium)=2.21e-6*avdep(ium).^2-6.6e-4*avdep(ium)+0.075;

dr(mi:end)=rsc(mi:end).*((vs(mi:end)-4.5)/4.5)*3.4268;
rho(mi:end)=3.4268+dr(mi:end);
rho=round(rho*1000)/1000; vp=round(vp*1000)/1000;

all=[vs;vp;t;avdep;rho;rsc;dep];
end
