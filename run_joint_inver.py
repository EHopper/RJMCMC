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

pr = cProfile.Profile()
pr.enable()


#def try_running():
max_it=200000
rnd_sd = 10

save_name = 'MBEY_Ps'
rf_obs, swd_obs, all_lims = input_data.LoadObservations()

while os.path.exists('output/'+save_name):
    save_name += '_'+str(rnd_sd)
os.mkdir('output/'+save_name)
shutil.copyfile('input_data.py','output/'+save_name+'/input_data.py')
save_name = 'output/'+save_name+'/'+save_name
out = pipeline.JointInversion(rf_obs, swd_obs, all_lims, max_it, rnd_sd,
                              save_name, 'Ps')

pr.disable()
s=open(save_name+'/profiletimes.txt','w')
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
s.close()
