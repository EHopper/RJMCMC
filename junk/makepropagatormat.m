%(nn,tleng, alpan,betan,rhon,c, alpam,betam,rhom, thikm, nl, iwave)
clearvars
addpath('C:\Users\Emily\OneDrive\Documents\WORK\MATLAB\Scattered_Waves\RJMCMC\junk\');
vs=[1.2, 3.5, 4.4, 4.7]; idep=[23, 35, 60, 100]+1; 
[vp,rho,thick] = makevelmodel(vs,idep);
vp = round(vp*1e3)/1e3; rho=round(rho*1e3)/1e3;

nn=1025; tleng=102.4; c = 1/0.0618;
alpan=vp(end); alpam=vp(1:end-1);
betan=vs(end); betam=vs(1:end-1);
rhon=rho(end); rhom=rho(1:end-1);
thikm=thick(1:end-1);
nl=length(vs);

nsub1=nl-1;
apnsq=alpan.^2;
btnsq=betan.^2;
csq=c.^2;
ralpn=sqrt(csq/apnsq-1.0);
rbetn=sqrt(csq/btnsq-1.0);
gaman=2.*btnsq/csq;
cimag = complex(0.,1.);

%.....construction of stress-displacement vector transform matrix (e).
e=zeros(4,4);
a=e;
t=e;
e(1,1)=-2.0*btnsq/apnsq;
e(1,2) = 0;
e(1,3)=1./(rhon*apnsq);
e(1,4)=0;
e(2,1)=0;
e(2,2)=csq*(gaman-1.)/(apnsq*ralpn);
e(2,3)=0;
e(2,4)=1./(rhon*apnsq*ralpn);
e(3,1)=(gaman-1.)/(gaman*rbetn);
e(3,2)=0;
e(3,3)=-1./(rhon*csq*gaman*rbetn);
e(3,4)=0;
e(4,1)=0;
e(4,2)=1.;
e(4,3)=0;
e(4,4)=1./(rhon*csq*gaman);

for i=35+1%:350%tleng/0.25+1%   %loop over frequency  use freq(i) etc. (200)
   freq=(i-1)/tleng;
   omega=2*pi*freq;
   pk=omega/c; pks(i)=pk;
   for j=1:nsub1      % loop over layer (100)
      m = nl - j;
      apmsq=alpam(m).^2;
      btmsq=betam(m).^2;
      ralpm=sqrt(csq/apmsq-1.);
      rbetm=sqrt(csq/btmsq-1.);
      gamam=2.*btmsq/csq;
      gamm1=gamam-1.;
      gamsq=gamam.^2;
      gm1sq=gamm1.^2;
      rocsq=rhom(m)*csq;
      pm=pk*ralpm*thikm(m);
      qm=pk*rbetm*thikm(m);
      sinpm=sin(pm);
      sinqm=sin(qm);
      cospm=cos(pm);
      cosqm=cos(qm);

      vars={'ralpm','rbetm'};
      for kk = 1:length(vars)
          eval([vars{kk} '= round(' vars{kk} '*1e16)/1e16;']);
      end

%.....construction of progator matrix (a) from bottom to top.

      a(1,1)=gamam*cospm-gamm1*cosqm;
      a(1,2)=cimag*(gamm1*sinpm/ralpm+gamam*rbetm*sinqm);
      a(1,3)=-(cospm-cosqm)/rocsq;
      a(1,4)=cimag*(sinpm/ralpm+rbetm*sinqm)/rocsq;
      a(2,1)=-cimag*(gamam*ralpm*sinpm+gamm1*sinqm/rbetm);
      a(2,2)=-gamm1*cospm+gamam*cosqm;
      a(2,3)=cimag*(ralpm*sinpm+sinqm/rbetm)/rocsq;
      a(2,4)=a(1,3);
      a(3,1)=rocsq*gamam*gamm1*(cospm-cosqm);
      a(3,2)=cimag*rocsq*(gm1sq*sinpm/ralpm+gamsq*rbetm*sinqm);
      a(3,3)=a(2,2);
      a(3,4)=a(1,2);
      a(4,1)=cimag*rocsq*(gamsq*ralpm*sinpm+gm1sq*sinqm/rbetm);
      a(4,2)=a(3,1);
      a(4,3)=a(2,1);
      a(4,4)=a(1,1);

      %if (j - 1)  40, 40, 50
      if ((j-1)<=0)     %(40) ...first propagator matrix (a), layer index (n-1).
          t=e*a;
      else             % (50) .....continuous propagator matrix multiplication, i.e., (p) = (t)*(a).
          p=t*a;
          t=p;
          p=zeros(size(p));
      end
   end  
end