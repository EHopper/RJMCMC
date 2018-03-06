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
max_it=500
rnd_sd = 10


deps = np.concatenate((np.arange(0,10,0.2), np.arange(10,60,1), np.arange(60,201,5)))
model = pipeline.Model(vs = np.array([3.4, 4.5]), all_deps = deps,
                       idep = np.array([60, 120]),  
                       std_rf = 0, lam_rf = 0, std_swd = 0)

rf_obs = pipeline.SynthesiseRF(pipeline.MakeFullModel(model))
swd_obs = pipeline.SynthesiseSWD(pipeline.MakeFullModel(model), 1/np.arange(0.02,0.1, 0.01))
all_lims = pipeline.Limits(
        vs = (0.5,5.5), dep = (0,200), std_rf = (0,0.05),
        lam_rf = (0.05, 0.5), std_swd = (0,0.15))

out = pipeline.JointInversion(rf_obs, swd_obs, all_lims, max_it, rnd_sd)

actual_model = pipeline.SaveModel(pipeline.MakeFullModel(model),out[1][:,0])

plt.figure(); plt.title("Suite of Velocity models")
for k in range(out[1][1,].size-2): 
    colstr = str(0.75-k/2/out[1][1,].size)
    plt.plot(out[1][:,k+1],out[1][:,0],
          '-',linewidth=1,color=colstr)
plt.gca().invert_yaxis()
plt.plot(actual_model,out[1][:,0],'r-',linewidth=3)
plt.plot(out[1][:,-1],out[1][:,0],'c-',linewidth=1)

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
s=open('thingy2.txt','w')
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()