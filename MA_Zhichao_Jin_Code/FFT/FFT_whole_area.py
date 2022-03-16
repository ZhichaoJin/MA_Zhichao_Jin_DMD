# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:36:50 2020

@author: lerch
"""

import numpy as np
import pandas as pd
import plot_snapshots as plts
import functions_fft as ffc

fname = r"\\nas.tu-clausthal.de\win-home$\zj19\Desktop\MA\snapshotbase_Shroud_2.csv"
delim = " "

dt = 5.859*10**(-5) # time step of signal, 4000 rev/min with 256 steps
L = 0.015 # seconds covered by data, 4000 U/min, and 1 rev taken
t = np.arange(0,L,dt)

N = np.size(t) # number of samples
n = int(N/2) # size of FFT results (number of calculated frequencies)

A = pd.read_csv(fname, delim, header = None).values
rowsA = len(A)
X_power = np.zeros((rowsA, n)) # will contain FFT results in every row for a different point

Fs = 1/dt #sampling frequency
array = np.arange(n)
freqs = np.real(Fs*array/N)
for i in range (rowsA): # iterate over all rows of the snapshotbase (point by point)
    A_point = A[i,:] # for creating a point data
    #FFT
    freqs, xpower = ffc.fourier_abs(A_point,dt,t)
    X_power[i,:] = xpower

np.savetxt("X_power_whole2.csv", X_power)
np.savetxt("freqs.csv", freqs)
# G = pd.read_csv("geometry_2.csv", skiprows = 0, delimiter = " ", names=["X","Y","Z"])
# X = G["X"].values
# Y = G["Y"].values
# Z = G["Z"].values
    
# plts.plotFFT_alex(X,Y,Z,X_power,freqs)
