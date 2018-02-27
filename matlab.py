# -*- coding: utf-8 -*-
"""

Matlab compatibility layer. Do things the way matlab arbitrarily chooses to do.

Created on Mon Feb 19 16:15:45 2018

@author: emily
"""

from scipy import signal
from numpy import linalg

def filtfilt(b, a, data):
    return signal.filtfilt(b, a, data, padlen=3*(max(len(a), len(b)) - 1))

def mldivide(a, b, data):
    # This works as long as the rank is the same as the number of num vars
    if linalg.matrix_rank(a) == a.shape[1]:
        return linalg.lstsq(a, b)[0]
    # from https://stackoverflow.com/questions/33559946/numpy-vs-mldivide-matlab-operator
#    else:
#        from itertools import combinations
#        for nz in combinations(range(num_vars), rank):    # the variables not set to zero
#            try: 
#                sol = np.zeros((num_vars, 1))  
#                sol[nz, :] = np.asarray(np.linalg.solve(A[:, nz], b))
#            except np.linalg.LinAlgError:     
#                pass
    