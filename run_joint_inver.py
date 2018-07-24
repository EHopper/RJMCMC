# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 14:23:28 2018

@author: emily
"""

import pipeline
import os
import input_data
import pstats
import cProfile
import shutil
import matplotlib.pyplot as plt
import numpy as np

pr = cProfile.Profile()
pr.enable()



max_it = int(1e6)#250000
rnd_sd = 10# random seed


save_name = 'KIPE_Both'



rf_obs, swd_obs, all_lims, vs = input_data.LoadObservations()
plt.figure(figsize = (14,8))
npan = 1+len(rf_obs)
for irf in range(len(rf_obs)):
    plt.subplot(1, npan, irf+1); plt.title('Receiver Function')
    rft = np.arange(0,rf_obs[irf].dt*rf_obs[irf].amp.size,rf_obs[irf].dt)
    plt.plot(rf_obs[irf].amp, rft, 'r-', linewidth=2)
    plt.plot(rf_obs[irf].amp-rf_obs[irf].std, rft, 'r--', linewidth=1)
    plt.plot(rf_obs[irf].amp+rf_obs[irf].std, rft, 'r--', linewidth=1)
    plt.plot([0,0],[0,30],'--',color='0.6')
    plt.ylim(30,0)
    plt.xlabel('RF Amplitude')
    plt.ylabel('Time (s)')

plt.subplot(1, npan, npan); plt.title('Surface Wave Dispersion')
plt.plot(swd_obs.period, swd_obs.c,  'r-', linewidth=2)
plt.xlabel('Period (s)')
plt.ylabel('Phase velocity (km/s)')
plt.tight_layout()


#%%
save_name += '_%d' % rnd_sd  # % is printf() type function
suffix = 0
def outdir_fn():
    return os.path.join('output', '%s_%05d' % (save_name, suffix))

while os.path.exists(outdir_fn()):
    suffix += 1

outdir = outdir_fn()

os.mkdir(outdir)
shutil.copyfile('input_data.py', os.path.join(outdir, 'input_data.py'))


out = pipeline.JointInversion(rf_obs, swd_obs, all_lims, max_it, rnd_sd,
                              os.path.join(outdir, save_name))



pr.disable()
s=open(os.path.join(outdir, 'profiletimes.txt'), 'w')
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
s.close()
