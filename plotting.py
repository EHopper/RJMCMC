# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 07:33:21 2018

@author: emily
"""

save_name = 'NKAL_Sp_scale5_10'#'MBEY_Sp_scale5_10'#'MBEY_Ps_scale5'
save_every = 100

import pipeline
import numpy as np
import matplotlib.pyplot as plt
import shutil

shutil.copyfile('./output/'+save_name+'/input_data.py', './input_data.py')

import input_data

rf_obs, swd_obs, all_lims = input_data.LoadObservations()

save_name = 'output/'+save_name+'/'+save_name
all_models = np.load(save_name+'_AllModels.npy')

good_mods = all_models[:,np.where(all_models[0,]>0)[0]]
nit = good_mods.shape[1]
nit_cutoff = int(nit/5)
good_mods = good_mods[:,-nit_cutoff:]
mean_mod = np.mean(good_mods, axis = 1)
std_mod = np.std(good_mods, axis = 1)

good_mod = pipeline.Model(vs = mean_mod, all_deps = all_models[:,0],
                          idep = np.arange(0,mean_mod.size),
                          lam_rf = 0, std_rf = 0, std_swd = 0)
fullmodel = pipeline.MakeFullModel(good_mod)



fig1 = plt.figure(figsize = (14,8));

ax1 = plt.subplot(131)
for k in range(all_models[1,].size-1):
    colstr = str(0.75-k/2/all_models[1,].size)
    ax1.plot(all_models[:,k],all_models[:,0],
          '-',linewidth=1,color=colstr)
ax1.set_ylim((195,0))
#ax1.plot(actual_model,all_models[:,0],'r-',linewidth=3)
ax1.set_xlim((2,5.5))
ax1.set_xlabel('Shear Velocity (km/s)')
ax1.set_ylabel('Depth (km)')
ax1.set_title("{} iterations".format(nit*save_every))

ax3 = plt.subplot(132)
for k in range(good_mods[0,].size-1):
    colstr = str(0.85-k/2/good_mods[0,].size)
    ax3.plot(good_mods[:,k],all_models[:,0],
          '-',linewidth=1,color=colstr)
ax3.plot(mean_mod,all_models[:,0],'b-',linewidth = 2)
ax3.plot(mean_mod+std_mod, all_models[:,0],'c-',linewidth = 1)
ax3.plot(mean_mod-std_mod, all_models[:,0],'c-',linewidth = 1)
#ax3.plot(actual_model,all_models[:,0],'r--',linewidth=1)
ax3.set_xlim((2,5.5))
ax3.set_ylim((195,0))
ax3.set_xlabel('Shear Velocity (km/s)')
ax3.set_ylabel('Depth (km)')
ax3.set_title('Most recent {}'.format(good_mods.shape[1]))

ax4 = plt.subplot(133)
for k in range(good_mods[0,].size-1):
    colstr = str(0.85-k/2/good_mods[0,].size)
    ax4.plot(good_mods[:,k],all_models[:,0],
          '-',linewidth=1,color=colstr)
ax4.plot(mean_mod,all_models[:,0],'b-',linewidth = 2)
ax4.plot(mean_mod+std_mod, all_models[:,0],'c-',linewidth = 1)
ax4.plot(mean_mod-std_mod, all_models[:,0],'c-',linewidth = 1)
#ax3.plot(actual_model,all_models[:,0],'r--',linewidth=1)
ax4.set_xlim((3.0,4.7))
ax4.set_ylim((60,0))
ax4.set_xlabel('Shear Velocity (km/s)')
ax4.set_ylabel('Depth (km)')
ax4.set_title('Most recent {}'.format(good_mods.shape[1]))




plt.tight_layout()



allvels = np.arange(all_lims.vs[0],all_lims.vs[1],0.01)
evendeps = np.arange(0,all_models[-1,0],0.1)
i_ed = np.zeros(evendeps.shape, dtype = int)
for k in range(all_models[:,0].size-1,0,-1):
    i_ed[all_models[k,0]>=evendeps] = k

mod_space = np.zeros((evendeps.size,allvels.size))
for k in range(1,good_mods.shape[1]):
    even_vels = good_mods[i_ed,-k]
    inds = np.round(even_vels-all_lims.vs[0],2)/0.01-1
    inds = inds.astype(int)
    mod_space[range(mod_space.shape[0]),inds] += 1



synth_swd = pipeline.SynthesiseSWD(fullmodel, swd_obs.period, 1e6)
if nit*save_every<1e5:
    synth_swd_coarse = pipeline.SynthesiseSWD(fullmodel, swd_obs.period, 1)
synth_rf = pipeline.SynthesiseRF(fullmodel, rf_obs)


plt.figure(figsize = (14,8));

ax2=plt.subplot(131)
ax2.imshow(np.log10(mod_space[-1::-1]+1e-1), cmap = 'viridis', aspect = allvels[-1]/evendeps[-1],
           extent = [allvels[0], allvels[-1], evendeps[0], evendeps[-1]])
ax2.invert_yaxis()
ax2.set_xlabel('Shear Velocity (km/s)')
ax2.set_ylabel('Depth (km)')
ax2.xaxis.set_label_position('top')
ax2.xaxis.tick_top()
ax2.set_xlim((1.5,5))


plt.subplot(132); plt.title('Receiver Function\n real: red; synth: grey')
rft = np.arange(0,rf_obs.dt*rf_obs.amp.size,rf_obs.dt)
plt.plot(rf_obs.amp, rft, 'r-', linewidth=2)
plt.plot(synth_rf.amp,rft, '-',color = '0.25', linewidth=2)
plt.plot(rf_obs.amp-0.02,rft, 'r--', linewidth=1)
plt.plot(rf_obs.amp+0.02,rft, 'r--', linewidth=1)
plt.plot([0,0],[0,30],'--',color='0.6')
plt.ylim(30,0)
plt.xlabel('RF Amplitude')
plt.ylabel('Time (s)')

plt.subplot(133); plt.title('Surface Wave Dispersion\n real: red; synth: grey')
plt.plot(swd_obs.period, swd_obs.c,  'r-', linewidth=2)
plt.plot(synth_swd.period, synth_swd.c, '-',color = '0.25', linewidth=2)
if nit*save_every<1e5:
    plt.plot(synth_swd_coarse.period, synth_swd_coarse.c, '-',color = '0.5', linewidth=2)
plt.plot(swd_obs.period, swd_obs.c-0.025,  'r--', linewidth=1)
plt.plot(swd_obs.period, swd_obs.c+0.025,  'r--', linewidth=1)
plt.xlabel('Period (s)')
plt.ylabel('Phase velocity (km/s)')
plt.tight_layout()

#%%

misfits = np.load(save_name+'_Misfit.npy')
nm = misfits.shape[1]#int(misfits.size/3)
# plot vertical lines where increase number of permissable layers
inc_ints = 750*np.arange(1,15)*np.arange(2,16)
good_it = nit - nit_cutoff


plt.figure(figsize = (12,5))
plt.title("Mahalanobis distance")
plt.plot((misfits[0,]));
plt.ylim(0.9*(np.min(misfits[0,10:])),
         1.1*(np.max(misfits[0,10:])))
for k in range(inc_ints.size):
    plt.plot(inc_ints[[k,k]]/save_every,[0,100],'--',color = '0.6')
plt.plot(good_it*np.ones(2),[0,100], 'r--')
plt.ylabel('Least Squares Misfit (phi)')
plt.xlabel('Iteration # /100')
plt.xlim(0,nm)

plt.figure(figsize = (10,5))
plt.subplot(121)
inds = np.where(misfits[1,]<3e3)[0]
plt.title("Likelihood of accepting new model")
plt.plot(np.log10(misfits[1,inds]));
plt.ylim(0.9*np.log10(np.min(misfits[1,inds])),
         1.1*np.log10(np.max(misfits[1,inds])))
plt.plot(good_it*np.ones(2),[0,100], 'r--')
plt.plot([0, nm],np.log10([1,1]),'--',color = '0.6')
plt.ylabel('log10(alpha(m|m0))')
plt.xlabel('Iteration # /100')
plt.xlim(0,nm)

plt.subplot(122); plt.title('Acceptance Rate')
plt.plot(misfits[2,]*100)
plt.plot([0,nm],[40,40],'--',color = '0.6')
plt.plot(good_it*np.ones(2),[20,70], 'r--')
plt.ylabel('Acceptance Rate (%)')
plt.xlabel('Iteration # /100')
plt.xlim(0,nm)
plt.tight_layout()

#%%
hyperparams = np.load(save_name+'_Hyperparams.npy')
plt.figure(figsize = (12,14))
plt.subplot(311)
plt.title('RF Noise standard deviation')
plt.plot(hyperparams[0:nm,0])
plt.xlabel('Iteration # /100')
plt.ylabel('Std of RF')
plt.xlim(0,nm)
plt.ylim(0.04,0.0525)
for k in range(inc_ints.size):
    plt.plot(inc_ints[[k,k]]/save_every,[0.04,0.06],'--',color = '0.6')
plt.plot(good_it*np.ones(2),[0.04, 0.06], 'r--')

plt.subplot(312)
plt.title('RF Noise Correlation')
plt.plot(hyperparams[0:nm,1])
plt.xlabel('Iteration # /100')
plt.ylabel('Lambda of RF')
plt.xlim(0,nm)
plt.ylim(0.1, 0.525)
for k in range(inc_ints.size):
    plt.plot(inc_ints[[k,k]]/save_every,[0.,0.6],'--',color = '0.6')
plt.plot(good_it*np.ones(2),[0., 0.6], 'r--')

plt.subplot(313)
plt.title('SWD Noise standard deviation')
plt.plot(hyperparams[0:nm,2])
plt.xlabel('Iteration # /100')
plt.ylabel('Std of SWD')
plt.xlim(0,nm)
plt.ylim(0.05,0.16)
for k in range(inc_ints.size):
    plt.plot(inc_ints[[k,k]]/save_every,[0.05, 0.175],'--',color = '0.6')
plt.plot(good_it*np.ones(2),[0.05,0.175], 'r--')
plt.tight_layout()

