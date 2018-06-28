# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 07:31:01 2018

@author: emily
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 07:31:01 2018

@author: emily
"""

import pipeline
import numpy as np




def LoadObservations():

	rf_obs = pipeline.RecvFunc(
				amp = np.array([0.056, 0.059, 0.056, 
                       0.040, 0.042, 0.014, -0.003, -0.002, 0.010, 
                       0.010, 0.009, 0.016, 0.025, 0.018, -0.003, 
                       -0.003, 0.025, 0.091, 0.129, 0.091, 0.017, 
                       -0.011, -0.006, 0.014, 0.034, 0.009, -0.030, 
                       -0.027, -0.021, -0.001, -0.004, -0.009, -0.016, 
                       -0.019, -0.014, 0.000, 0.014, 0.009, -0.001, 
                       0.001, 0.009, -0.000, -0.014, -0.016, -0.014, 
                       -0.002, -0.002, -0.003, -0.002, -0.006, -0.011, 
                       -0.005, -0.009, -0.007, 0.002, -0.001, -0.013, 
                       0.003, 0.027, 0.047, 0.026, -0.006, -0.036, 
                       -0.024, -0.013, 0.003, 0.009, -0.005, -0.015, 
                       -0.011, -0.006, -0.005, -0.010, -0.018, -0.035, 
                       -0.023, -0.015, -0.019, -0.015, -0.012, -0.027, 
                       -0.045, -0.021, 0.006, 0.023, 0.011, 0.013, 
                       -0.009, -0.036, -0.023, -0.020, -0.024, -0.022, 
                       -0.007, -0.004, 0.016, 0.034, 0.019, -0.019, 
                       -0.043, -0.031, -0.014, 0.005, 0.002, 0.010, 
                       0.004, -0.005, -0.001, 0.000, -0.007, -0.013, 
                       -0.000, -0.008, -0.018, -0.014, -0.019, -0.006, 
                       0.009, 0.010, -0.001
					]),
				dt = 0.250, ray_param = 0.06147,
				std_sc = 5, rf_phase = 'Ps',
				filter_corners = [1, 100]
				)

	swd_obs = pipeline.SurfaceWaveDisp(
				period = np.array([9.000, 10.120, 11.570, 
                       13.500, 16.200, 20.250, 25.000, 32.000, 40.000, 
                       50.000, 60.000, 80.000
					]),
				c = np.array([3.2200, 3.2860, 3.3290, 
                       3.3880, 3.4830, 3.5980, 3.8230, 3.9690, 4.0330, 
                       4.0410, 4.1230, 4.1100
					])
				)

	all_lims = pipeline.Limits(
				vs = (0.5, 5.0),dep = (0,200), std_rf = (0, 0.05),
				lam_rf = (0.05, 0.5), std_swd = (0, 0.05), crustal_thick = (25,))

	return (rf_obs, swd_obs, all_lims)