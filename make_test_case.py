# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 21:27:39 2018

@author: Emily
"""

import pipeline
import numpy as np
import os

rf_phase = 'Both' #'Sp' # 'Ps' # 'Both'
weight_by = 'rf4' # 'even' 'rf2'

basedir = 'D:/Scattered_Waves/RJMCMC/'
#'C:/Users/Emily/Documents/RJMCMC/'
savename = open(os.path.join(basedir, 'input_data.py'), mode = 'w')

deps = np.concatenate((np.arange(0,10,0.2), np.arange(10,60,1),
                       np.arange(60,201,5)))

model = pipeline.Model(vs = np.array([3.7, 4.3, 4.7, 4.6]),
                       all_deps = deps, idep = np.array([10, 30, 90, 110]),
                       std_rf_sc = 1, lam_rf = 0.2, std_swd_sc = 1)

fullmodel = pipeline.MakeFullModel(model)
savemodel = pipeline.SaveModel(fullmodel, deps)

stdsw = np.array([0.7880, 0.3990, 0.5520,
                       0.4860, 0.3620, 0.2210, 0.0070, 0.0120, 0.0130,
                       0.0160, 0.0150, 0.0100
					])
swd_in = pipeline.SurfaceWaveDisp(
            period = np.array([9, 10.12, 11.57, 13.5, 16.2, 20.25, 25,
                           32, 40, 50, 60, 80,]),
            c = np.zeros(12,), std = stdsw, #0.01*np.ones(12,)
            )
swd = pipeline.SynthesiseSWD(fullmodel, swd_in, 1e6)
stdrf = np.array([0.012, 0.026, 0.033,
                       0.039, 0.043, 0.045, 0.047, 0.042, 0.037,
                       0.042, 0.045, 0.043, 0.042, 0.041, 0.041,
                       0.038, 0.038, 0.038, 0.037, 0.036, 0.035,
                       0.034, 0.032, 0.029, 0.025, 0.022, 0.019,
                       0.017, 0.016, 0.015, 0.013, 0.012, 0.013,
                       0.013, 0.014, 0.013, 0.012, 0.012, 0.012,
                       0.011, 0.011, 0.011, 0.013, 0.016, 0.021,
                       0.026, 0.030, 0.031, 0.035, 0.036, 0.033,
                       0.029, 0.023, 0.016, 0.014, 0.016, 0.017,
                       0.016, 0.017, 0.019, 0.019, 0.019, 0.018,
                       0.018, 0.018, 0.018, 0.018, 0.015, 0.015,
                       0.011, 0.010, 0.014, 0.019, 0.022, 0.021,
                       0.020, 0.019, 0.019, 0.024, 0.034, 0.047,
                       0.059, 0.068, 0.076, 0.077, 0.073, 0.068,
                       0.058, 0.046, 0.034, 0.024, 0.022, 0.032,
                       0.043, 0.054, 0.062, 0.065, 0.066, 0.063,
                       0.059, 0.059, 0.068, 0.078, 0.084, 0.088,
                       0.085, 0.081, 0.073, 0.060, 0.049, 0.037,
                       0.036, 0.037, 0.038, 0.040, 0.040, 0.041,
                       0.043, 0.045, 0.047])
rf_in = [pipeline.RecvFunc(amp = np.array([0,1]), dt = 0.25,
                           std = stdrf, #0.02*np.ones(120,),
                          rf_phase = 'Ps', ray_param = 0.06147,
                          filter_corners = [1,100], std_sc = 0.5,
                          weight_by = 'even')]


rf_Ps = pipeline.SynthesiseRF(fullmodel, rf_in)
rf_Sp = pipeline.SynthesiseRF(fullmodel, [rf_in[0]._replace(rf_phase = 'Sp',
                                                       ray_param = 0.11012,
                                                       filter_corners = [4,100],
                                                       std_sc = 5)])

if rf_phase == 'Ps': rf = rf_Ps
if rf_phase == 'Sp': rf = rf_Sp
if rf_phase == 'Both': rf = [rf_Ps[0], rf_Sp[0]]

print('\n\nimport pipeline\nimport numpy as np\n',
      '\ndef LoadObservations():\n',
      '\n\tswd_obs = pipeline.SurfaceWaveDisp(',
      '\n\t\tperiod = np.array([', end = '', sep = '', file = savename)
for k in range(swd.period.size):
    print(swd.period[k], ', ', end = '', sep ='', file = savename)
    if not k%6: print('\n\t\t\t', end = '', sep = '', file = savename)
print('\n\t\t\t]),\n\n\t\tc = np.array([',end = '', sep = '', file = savename)
for k in range(swd.c.size):
    print(round(swd.c[k],3), ', ', end = '', sep = '', file = savename)
    if not k%6: print('\n\t\t\t', end = '', sep = '', file = savename)
print('\n\t\t\t]),\n\n\t\tstd = np.array([',end = '', sep = '', file = savename)
for k in range(swd.std.size):
    print(round(swd.std[k],3), ', ', end = '', sep = '', file = savename)
    if not k%6: print('\n\t\t\t', end = '', sep = '', file = savename)
print('\n\t\t\t]),\n\t\t)',end = '', sep = '', file = savename)

print('\n\tall_lims = pipeline.Limits(\n\t\t\tvs = (0.5, 5.0),',
        'dep = (0,200), std_rf_sc = (1, 2), \n\t\t\tlam_rf = (0.05, 0.5),',
        'std_swd_sc = (1, 2), \n\t\t\tcrustal_thick = (25,)\n\t\t\t)',
        end = '', sep = '', file = savename)


print('\n\trf_obs = [', end = '', sep = '', file = savename)
for irf in range(len(rf)):
    print('\n\t\tpipeline.RecvFunc(',
            '\n\t\tamp = np.array([', end = '', sep = '', file = savename)
    for k in range(rf[irf].amp.size):
        print(round(rf[irf].amp[k],3), ', ', end = '', sep = '', file = savename)
        if not k%6: print('\n\t\t\t', end = '', sep = '', file = savename)
    print('\n\t\t\t]),\n\t\tstd = np.array([', end = '', sep = '', file = savename)
    for k in range(rf[irf].std.size):
        print(round(rf[irf].std[k],3), ', ', end = '', sep = '', file = savename)
        if not k%6: print('\n\t\t\t', end = '', sep = '', file = savename)
    print('\n\t\t\t]), \n\n\t\t\tdt = ', rf[irf].dt, ', ray_param = ', rf[irf].ray_param, ',',
        'std_sc = ', rf[irf].std_sc, ',\n\t\t\trf_phase = \'', rf[irf].rf_phase, '\'',
        ', filter_corners =  ', rf[irf].filter_corners, ','
        '\n\t\t\tweight_by = \'', weight_by, '\'\n\t\t),', end = '', sep = '',
        file = savename)
print(']', end = '', sep = '', file = savename)

print('\n\n\tvs_in = np.array([', end = '', file = savename)
for k in range(savemodel.size):
    print('[', deps[k], ', ', round(savemodel[k],3), '], ', end = '', sep = '',
            file = savename)
    if not k%4: print('\n\t\t', end = '', sep = '', file = savename)
print('\n\t\t])', end = '', sep = '', file = savename)

print('\n\n\treturn (rf_obs, swd_obs, all_lims, vs_in)', file = savename)

open(os.path.join(basedir, 'input_data.py'), mode = 'r')
