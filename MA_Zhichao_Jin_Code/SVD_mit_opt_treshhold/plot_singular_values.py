#!/usr/bin/env python3sftp://niiksl12@blogin.hlrn.de/gfs1/work/niiksl12/DMD/nrot_vel25/create_matrix_velocity.py

# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 12:11:09 2018

@author: ksl12
"""

from re import X
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plotSV(scale, saveas, lines):
    plt.figure()
    plt.yscale(scale)
    plt.plot(x, S,label='sigma', color = 'b')
    plt.axvline(x = lines[0], label='s=%i : 99 percent' %lines[0], color = 'r')
    plt.axvline(x = lines[1], label='s=%i : Opt Truncation' %lines[1], color = 'g')
    plt.axvline(x = lines[2], label='s=%i : 90 percent' %lines[2], color = 'b')
    plt.legend(loc="upper center")
    plt.xlabel("number of singular values")
    plt.ylabel("magnitude of singular values")
    plt.savefig(saveas, bbox_inches='tight') #tight, so the y axis label is not cut off

S = pd.read_csv(r"\\nas.tu-clausthal.de\win-home$\zj19\Desktop\optdmd\S.csv", delimiter = " ", header = None).values
A = pd.read_csv(r"\\nas.tu-clausthal.de\win-home$\zj19\Desktop\MA\snapshotbase_Shroud_2.csv", delimiter = " ", header = None).values
m = np.size(A, 1) #columns of A

X = A[:,:m-1]
S_plot = S[1:]

##Berechnung des optimalen Treshhold
omega = lambda x: 0.56 * x**3 - 0.95 * x**2 + 1.82 * x + 1.43
beta = np.divide(*sorted(X.shape))
tau = np.median(S) * omega(beta)
rank = np.sum(S > tau)

e_max = np.sum(S_plot)
e_99 = 0.99*e_max
e_90 = 0.9*e_max

e_kum = 0
i = 0
while e_kum < e_99:
    e_kum = e_kum+S_plot[i]
    i = i+1
    
print ("e_99 bei i = %i" %i)

e_kum = 0
k = 0
while e_kum < e_90:
    e_kum = e_kum+S_plot[k]
    k = k+1

print ("e_90 bei k = %i" %k)



n = len(S)
x = np.linspace(0, n-1, n)

lines = ((i,rank,k))

plotSV("linear", "S_lin_2.png", lines)
plotSV("log", "S_log_2.png", lines)
