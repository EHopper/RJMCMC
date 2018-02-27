# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 14:45:35 2018

@author: emily
"""

def run():
    import numpy as np
    import matplotlib.pyplot as plt
    import pipeline

    # Sample rate and desired cutoff frequencies (in Hz).
    dt = 0.15
    long_T = 100
    short_T = 5

    # Filter a noisy signal.
    T = 20.
    t = np.arange(0, T, dt)
    x =  np.sin(1/4 *2 * np.pi * t)
    x += 0.5* np.cos(2 * np.pi * t )
    #x += a * np.cos(2 * np.pi * f0 * t + .11)
    #x += 0.03 * np.cos(2 * np.pi * 2000 * t)
    plt.figure(2)
    plt.clf()
    plt.plot(t, x, label='Noisy signal')

    y = pipeline.BpFilt(x, short_T, long_T, dt)
    plt.plot(t, y, label='Filtered signal')
    plt.xlabel('time (seconds)')
   # plt.hlines([-a, a], 0, T, linestyles='--')
   # plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    plt.show()


run()