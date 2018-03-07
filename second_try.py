# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 14:23:28 2018

@author: emily
"""

import pipeline
import numpy as np
import matplotlib.pyplot as plt
import pstats
import cProfile
 
pr = cProfile.Profile()
pr.enable()


#def try_running():
max_it=20000
rnd_sd = 1


deps = np.concatenate((np.arange(0,10,0.2), np.arange(10,60,1), np.arange(60,201,5)))
model = pipeline.Model(vs = np.array([1.8, 2.4, 3.4, 4.5, 4.7, 4.65]), all_deps = deps,
                       idep = np.array([10, 32, 41, 60, 96, 120]),  
                       std_rf = 0, lam_rf = 0, std_swd = 0)

rf_obs = pipeline.SynthesiseRF(pipeline.MakeFullModel(model))
swd_obs = pipeline.SynthesiseSWD(pipeline.MakeFullModel(model), 1/np.arange(0.02,0.1, 0.01))
all_lims = pipeline.Limits(
        vs = (0.5,5.5), dep = (0,200), std_rf = (0,0.05),
        lam_rf = (0.05, 0.5), std_swd = (0,0.15))

out = pipeline.JointInversion(rf_obs, swd_obs, all_lims, max_it, rnd_sd)

actual_model = pipeline.SaveModel(pipeline.MakeFullModel(model),out[1][:,0])
all_models = out[1]
#%%


all_models = np.load('testsave.npy')
good_mods = all_models[:,np.where(all_models[0,]>0)[0]]
nit = good_mods.shape[1]
good_mods = good_mods[:,-int(nit/6):]
mean_mod = np.mean(good_mods, axis = 1)
std_mod = np.std(good_mods, axis = 1)



fig1 = plt.figure();

ax1 = plt.subplot(121)
for k in range(all_models[1,].size-1): 
    colstr = str(0.75-k/2/all_models[1,].size)
    plt.plot(all_models[:,k],all_models[:,0],
          '-',linewidth=1,color=colstr)
ax1.invert_yaxis()
ax1.plot(actual_model,all_models[:,0],'r-',linewidth=3)
ax1.plot(all_models[:,-1],all_models[:,0],'c-',linewidth=1)
ax1.set_xlim((1.5,5))
ax1.set_xlabel('Shear Velocity (km/s)')
ax1.set_ylabel('Depth (km)')
ax1.set_title("{} iterations".format(nit*100))

ax3 = plt.subplot(122)
for k in range(good_mods[0,].size-1): 
    colstr = str(0.75-k/2/good_mods[0,].size)
    ax3.plot(good_mods[:,k],all_models[:,0],
          '-',linewidth=1,color=colstr)
ax3.invert_yaxis()
ax3.plot(mean_mod,all_models[:,0],'c-',linewidth = 2)
ax3.plot(mean_mod+std_mod, all_models[:,0],'c-',linewidth = 1)
ax3.plot(mean_mod-std_mod, all_models[:,0],'c-',linewidth = 1)
ax3.plot(actual_model,all_models[:,0],'r--',linewidth=1)
ax3.set_xlim((1.5,5))
ax3.set_xlabel('Shear Velocity (km/s)')
ax3.set_ylabel('Depth (km)')

ax3.set_title('Most recent')


allvels = np.arange(all_lims.vs[0],all_lims.vs[1],0.01)
evendeps = np.arange(0,all_models[-1,0],0.1)
i_ed = np.zeros(evendeps.shape, dtype = int)
for k in range(all_models[:,0].size-1,0,-1):
    i_ed[all_models[k,0]>=evendeps] = k
    
mod_space = np.zeros((evendeps.size,allvels.size))
for k in range(1,good_mods.shape[1]):
    even_vels = good_mods[i_ed,-k]
    inds = np.round(even_vels-all_lims.vs[0],2)/0.01
    inds = inds.astype(int)
    mod_space[range(mod_space.shape[0]),inds] += 1 

plt.tight_layout()

fig2 = plt.figure()
ax2 = plt.subplot(121)
ax2.imshow(np.log10(mod_space[-1::-1]+1e-1), cmap = 'viridis', aspect = allvels[-1]/evendeps[-1],
           extent = [allvels[0], allvels[-1], evendeps[0], evendeps[-1]])
ax2.invert_yaxis()
ax2.set_xlabel('Shear Velocity (km/s)')
ax2.set_ylabel('Depth (km)')
ax2.xaxis.set_label_position('top')
ax2.xaxis.tick_top()
ax2.set_xlim((1.5,5))

#mod_space_actual = np.zeros_like(mod_space)
#even_vels = actual_model[i_ed]
#inds = np.round(even_vels-all_lims.vs[0],2)/0.01
#mod_space_actual[range(mod_space.shape[0]),inds.astype(int)] += 1
#ax4 = plt.subplot(122)
#ax4.imshow(mod_space_actual[-1::-1], cmap = 'viridis', aspect = allvels[-1]/evendeps[-1]/2,
#           extent = [allvels[0], allvels[-1], evendeps[0], evendeps[-1]])
#ax4.set_xlabel('Shear Velocity (km/s)')
##ax4.set_ylabel('Depth (km)')
#ax4.invert_yaxis()
#ax4.xaxis.set_label_position('top')
#ax4.xaxis.tick_top()
#ax4.set_xlim((1.5,5))
#plt.tight_layout()


fig1.savefig('VelocityModelSuite.png')
fig2.savefig('VelocityModelSuite_heatmap.png')
#plt.colorbar()

#%%
plt.figure(); plt.title('Receiver Function - real: red; synth: grey')
rft = np.arange(0,rf_obs.dt*rf_obs.amp.size,rf_obs.dt)
plt.plot(rft, rf_obs.amp, 'r-', linewidth=2)
synth_rf = pipeline.SynthesiseRF(out[5])
plt.plot(rft,synth_rf.amp, '-',color = '0.25', linewidth=1)

synth_swd = pipeline.SynthesiseSWD(out[5], swd_obs.period)
plt.figure(); plt.title('Surface Wave Dispersion - real: red; synth: grey')
plt.plot(swd_obs.period, swd_obs.c,  'r-', linewidth=2)
plt.plot(synth_swd.period, synth_swd.c, '-',color = '0.25', linewidth=1)


plt.figure(); plt.title("Mahalanobis distance (least squares misfit - phi)")
plt.plot(np.log10(out[2]))

plt.figure(); plt.title("Likelihood of accepting new model - alpha(m|m0)")
plt.plot(np.log10(out[3]))

print(np.mean(out[4]))

pr.disable()
s=open('thingy4.txt','w')
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
s.close()