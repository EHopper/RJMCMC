# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:16:01 2018

@author: emily
"""

import pipeline




# Inputs are the filenames with the RF and the SWD observations
# For RFs, textfile should be two columns - time and amplitude
#    timestep should be 0.25 s  (maybe play with this)
#    want one stacked RF stretched appropriately for an incident wave at 60 deg
filename_rf='rf_filename_here.txt'
filename_swd='swd_filename_here.txt'

# This code is largely based on...
# Bea12:    Bodin et al., 2012, JGR, doi:10.1029/2011JB008560
# KL14:     Kolb and Lekic, 2014, GJI, doi:10.1093/gji/ggu079


# Run iteratively through the whole process
# (1) Suggest model updates
# (2) Choose whether to accept model updates
# (2a) Calculate forward model for RFs and for SWD
# (2b) Calculate likelihood of acceptance based on misfit
# (3) If model has not converged, repeat

import numpy as np

# Load in the observed data
f=open(filename_rf,'r')
d_obs_rf=f.read()
f=open(filename_swd,'r')
d_obs_swd=f.read()



# This is iterative as update the models
max_iter=10   # According to Kolb and Lekic, this should be about 2e6
           # Need about 5e5 iterations to converge, generally
           # and once converged, they iterate 1e6 more times to define posterior

vs_lims=(0.5,5.5)        # min and max Vs in km/s
dep_lims=(0,200)       # min and max depth points in km
std_rf_lims=(0,0.05)   # min and max standard deviation of RFs
std_swd_lims=(0,0.15)  # min and max standard deviation of SWD
lam_rf_lims=(0.05,0.5) # min and max lambda for RFs (noise correlation)
      
    
    