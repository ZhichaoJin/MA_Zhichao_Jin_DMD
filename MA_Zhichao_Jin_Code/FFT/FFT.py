# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:36:50 2020

@author: lerch
"""

import numpy as np
import pandas as pd
import plot_snapshots as plts
import matplotlib.pyplot as plt
import functions_fft as ffc

fname = r"\\nas.tu-clausthal.de\win-home$\\zj19\Desktop\\MA\snapshotbase_shroud_1.csv"
delim = " "

dt = 60/(4000*256)# time step of signal, 4000 rev/min with 256 steps
L = 60/4000 # seconds covered by data, 4000 U/min, and 1 rev taken
t = np.arange(0,L,dt)

N = np.size(t) # number of samples
n = int(N/2) # size of FFT results (number of calculated frequencies)

Fs = 1/dt # sampling frequency for FFT
array = np.arange(N/2)
freqs = np.real(Fs*array/N) # all frequencies for which FFT will calculate modes
np.savetxt("freqs.csv", freqs)
A = pd.read_csv(fname, delim, header = None).values

rowsA = len(A)
X_power = np.zeros((rowsA, n)) # will contain FFT results in every row for a different point

for i in range (rowsA): # iterate over all rows of the snapshotbase (point by point)
    A_point = A[i,:] # for creating a point data
    #FFT
    freqs, xpower = ffc.fourier_abs(A_point,dt,t)
    X_power[i,:] = xpower
    #xhat = np.fft.fft(A_point, axis = 0)
    #X_power[i,:] = abs(xhat[0:n])/N

np.savetxt("X_power_whole.csv", X_power)   

G = pd.read_csv("geometry1.csv", skiprows = 0,delimiter = " ", names=["X","Y","Z"])
X = G["X"].values
Y = G["Y"].values
Z = G["Z"].values


freqs = pd.read_csv( "freqs.csv", delimiter = " ", header = None).values   
plts.plotFFT_alex(X,Y,Z,X_power,freqs)

# FFT mean power spectrum

# rowsX_power = len(X_power)
# colsX_power = len(X_power[0])
# X_power_mean = np.zeros(colsX_power)

# for i in range (colsX_power):
#     X_power_mean[i] = sum(X_power[:,i])/rowsX_power
    
# plt.figure(figsize=(20,10))
# plt.title("Comparison between FFT and DMD on mean pressure of the flow around a cylinder at 20m/s inlet velocity")
# plt.xlim(0,5)
# plt.ylim(-2,35)
# plt.plot(freqs,X_power_mean, label= 'FFT on whole area', color = "red")
# plt.plot([1.699999],[30.641500], marker = "o", markersize = 6, color = "red")
# plt.annotate("x = 1.699999, y = 30.641500",xy = (1.699999,30.641500), xytext = (1.699999+0.05,30.641500-0.5))

# plt.legend()
# plt.xlabel("f [Hz]")
# plt.ylabel("p [Pa]")
# plt.savefig('spectrum_mean_p.png')
