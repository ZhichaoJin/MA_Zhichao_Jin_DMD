# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 08:57:20 2020

@author: lerch
"""

# v - sequence of values over time
# dt - time step
# t - time sequence according to v

import numpy as np

# fourier transform that yields absolute values
def fourier_abs(v,dt,t): 
    N = np.size(t) # number of samples
    n = int(N/2)
    xhat = np.fft.fft(v, axis = 0)
    xpower = abs(xhat[0:n])*2/N
    Fs = 1/dt #sampling frequency
    array = np.arange(n)
    freqs = np.real(Fs*array/N)
    
    return freqs, xpower
