# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 14:23:28 2018

@author: emily
"""

import pipeline_three
import os
import input_data
import pstats
import cProfile
import shutil

pr = cProfile.Profile()
pr.enable()



max_it=250000
rnd_sd = 10


save_name = 'TEST_Both_scale5'



rf_obs_Ps, rf_obs_Sp, swd_obs, all_lims = input_data.LoadObservations()

save_name += '_%d' % rnd_sd  # % is printf() type function
suffix = None
def outdir_fn():
    if suffix is None:
        return os.path.join('output', save_name)
    else:
        return os.path.join('output', '%s_%05d' % (save_name, suffix))

while os.path.exists(outdir_fn()):
    if suffix is None:
        suffix = 0
    else:
        suffix += 1

outdir = outdir_fn()

os.mkdir(outdir)
shutil.copyfile('input_data.py', os.path.join(outdir, 'input_data.py'))


out = pipeline_three.TripleInversion(rf_obs_Ps, rf_obs_Sp, swd_obs, all_lims,
                            max_it, rnd_sd, os.path.join(outdir, save_name))



pr.disable()
s=open(os.path.join(outdir, 'profiletimes.txt'), 'w')
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
s.close()
