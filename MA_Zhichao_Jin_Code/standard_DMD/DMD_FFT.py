# -*- coding: utf-8 -*-
"""
Created on Wed May 27 13:52:52 2020

@author: lerch
"""

# Std Library imports
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['text.usetex'] = True #use latex in text
import matplotlib.pyplot as plt
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})  #use latex also in labels
from matplotlib.ticker import MultipleLocator # to add minor ticks

# Local imports
import functions_dmd as dfc
import functions_fft as ffc

def readFile(fname, delim):
    A = pd.read_csv(fname , delimiter = delim, header = None).values # read the snapshotbase
    # insert check for correct snapshotbase structure
    return A

plt.rcParams.update({'font.size':24})





fname = r"\\nas.tu-clausthal.de\win-home$\\zj19\Desktop\\MA\\snapshotbase_shroud_2.csv"
delim = " "
dt = 60/(4000*256) # time step of signal, 4000 rev/min with 256 steps
L = 0.015 # seconds covered by data, 4000 U/min, and 1 rev taken
s = 256 # stack length for DMD 1D
r = 20 # truncation threshold for DMD 1D
fmax = 1000 # max freq on x axis
ymax = 550 # max pressure or velocity on y axis
label_y = r'$\displaystyle P$ [Pa]'

t = np.arange(0,L,dt)
N = np.size(t) # number of samples
n = int(N/2)



#mehere Umdrehungen DMD FFT vergleichen
freqs = pd.read_csv(r"\\nas.tu-clausthal.de\win-home$\zj19\Desktop\optdmd\freqs.csv", header = None,delimiter = " ").values
X_power_whole2 = readFile(r"\\nas.tu-clausthal.de\win-home$\zj19\Desktop\optdmd\X_power_whole2.csv", delim)

f_im1 = pd.read_csv(r"\\nas.tu-clausthal.de\win-home$\zj19\Desktop\MA\DMDfreqs_imaginary2.csv", header = None,delimiter = " ").values
p1= pd.read_csv(r"\\nas.tu-clausthal.de\win-home$\zj19\Desktop\MA\DMDpower2.csv", header = None,delimiter = " ").values

f_im2 = pd.read_csv(r"\\nas.tu-clausthal.de\win-home$\zj19\Desktop\optdmd\optDMDfreqs_imaginary2.csv", header = None,delimiter = " ").values
p2 = pd.read_csv(r"\\nas.tu-clausthal.de\win-home$\zj19\Desktop\optdmd\optDMDpower2.csv", header = None,delimiter = " ").values
p2_mean=pd.read_csv(r"\\nas.tu-clausthal.de\win-home$\zj19\Desktop\optdmd\optDMDpower_Phi2.csv", header = None,delimiter = " ").values

p2=np.mean(p2,axis=1)#Mittelwert von Ampulitude jeder Frequenz(2D bis 1D)

freqs =np.asanyarray(freqs).squeeze()#去除列属性 shape von (x,1) bis (x,)
f_im1 =np.asanyarray(f_im1).squeeze()#去除列属性
f_im2 =np.asanyarray(f_im2).squeeze()#去除列属性
p1=np.asanyarray(p1).squeeze()
p2=np.asanyarray(p2).squeeze()
p2_mean=np.asanyarray(p2_mean).squeeze()

X_power_mean2 = np.mean(X_power_whole2,axis=0)

plt.figure(figsize=(20, 12)) 
plt.subplot(2, 1, 1)
plt.bar(f_im1, p1, label = 'DMD U2', color = "g", width = 5)
plt.scatter(freqs,X_power_mean2, label= 'FFT U2', color = "blue", linewidth = 5)
plt.legend()
plt.xlim(0,1500)
plt.ylim(0,ymax )
plt.ylabel(label_y)
plt.title("Vergleich mit DMD, Opt und FFT auf U2")
plt.grid(True, which = 'both')
plt.subplot(2, 1, 2)
plt.scatter(freqs,X_power_mean2, label= 'FFT U2', color = "blue", linewidth = 5)
plt.bar(f_im2, p2, label = 'optDMD U2', color = "black", width = 5)

plt.legend()
plt.xlabel(r'$\displaystyle f$ [Hz]')
plt.ylabel(label_y)
plt.xlim(0,1500)
plt.ylim(0,ymax )
plt.grid(True, which = 'both')

plt.savefig('spectrum_U2_DMD_FFT_OPT1.png')
plt.show()

