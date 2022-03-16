# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 18:58:57 2018

@author: Kathrin
"""

# CHANGE PLOT TITLE AND FILE NAME!!
# plots snapshots of a matrix with snapshots in the columns

import matplotlib
matplotlib.rcParams['text.usetex'] = True #use latex in text
import matplotlib.pyplot as plt
plt.switch_backend('agg') #to fix invalid DISPLAY variable
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})  #use latex also in labels
import matplotlib.colors as colors
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=600):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'mycmap', cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plotPhase(X ,Y ,Z, A, freqs):
    m = A.shape
    triang = matplotlib.tri.Triangulation(X, Y)
    for i in range(0,m[1]):
        f, ax = matplotlib.pyplot.subplots(figsize = (8,8))
        plt.rcParams.update({'font.size':24})
        plt.rc('xtick', labelsize=24) 
        plt.rc('ytick', labelsize=24) 
        ax.set_aspect('equal')
        ax.tricontourf(X, Y, Z, 24)
        ax.set_xlabel(r'$\displaystyle x$ [m]')
        ax.set_ylabel(r'$\displaystyle y$ [m]')
        mode = i
        plt.title("opt DMD mode phase %i, $\displaystyle f$ = %4.3f Hz" %(mode,freqs[i]))
        tpc = ax.tripcolor(triang, A[:,i], vmin = -1, vmax = 1, shading='gouraud', cmap = 'twilight_r')
        cbar = f.colorbar(tpc)
        cbar.set_label(r'$\displaystyle\varphi$ [$\displaystyle \pi$ rad]')
        plt.savefig('opt_DMD_mode_phase%i.png' %mode)
        plt.close()

def plotAmplitude(X ,Y ,Z, A, freqs):
    cmap = plt.get_cmap('jet')
    new_cmap = truncate_colormap(cmap, 0.5, 1)
    m = A.shape
    triang = matplotlib.tri.Triangulation(X, Y)
    for i in range(0,m[1]):
        f, ax = matplotlib.pyplot.subplots(figsize = (8,8))
        plt.rcParams.update({'font.size':24})
        plt.rc('xtick', labelsize=24) 
        plt.rc('ytick', labelsize=24) 
        ax.set_aspect('equal')
        ax.tricontourf(X, Y, Z, 24)
        ax.set_xlabel(r'$\displaystyle x$ [m]')
        ax.set_ylabel(r'$\displaystyle y$ [m]')
        mode = i
        plt.title("opt DMD mode amplitude %i, $\displaystyle f$ = %4.3f Hz" %(mode,freqs[i]))
        tpc = ax.tripcolor(triang, A[:,i], vmin = 0, vmax = 600, shading='gouraud', cmap = new_cmap)
        cbar = f.colorbar(tpc)
        cbar.set_label(r'$\displaystyle\Delta p$ [Pa]')
        plt.savefig('opt_DMD_mode_amplitude%i.png' %mode)
        plt.close()

def plotAmplitude_alex(X ,Y ,Z, A, freqs):
    m = A.shape
    triang = matplotlib.tri.Triangulation(X, Y)
    for i in range(0,m[1]):
        f, ax = matplotlib.pyplot.subplots(figsize = (8,8))
        plt.rcParams.update({'font.size':24})
        plt.rc('xtick', labelsize=24) 
        plt.rc('ytick', labelsize=24) 
        ax.set_aspect('equal')
        ax.tricontourf(X, Y, Z, 24)
        ax.set_xlabel(r'$\displaystyle x$ [m]')
        ax.set_ylabel(r'$\displaystyle y$ [m]')
        mode = i
        plt.title("optDMD $\displaystyle f$ = %4.3f Hz" %(freqs[i]))
        tpc = ax.tripcolor(triang, A[:,i], vmin = 0, vmax = 400, shading='gouraud', cmap = 'jet')
        cbar = f.colorbar(tpc)
        cbar.set_label(r'$\displaystyle\Delta p$ [Pa]')
        plt.savefig('opt_DMD_mode%i.png' %mode)
        plt.close()
        
def plotIm(X ,Y ,Z, A, freqs):
    m = A.shape
    triang = matplotlib.tri.Triangulation(X, Y)
    for i in range(0,m[1]):
        f, ax = matplotlib.pyplot.subplots(figsize = (8,8))
        plt.rcParams.update({'font.size':24})
        plt.rc('xtick', labelsize=24) 
        plt.rc('ytick', labelsize=24) 
        ax.set_aspect('equal')
        ax.tricontourf(X, Y, Z, 24)
        ax.set_xlabel(r'$\displaystyle x$ [m]')
        ax.set_ylabel(r'$\displaystyle y$ [m]')
        mode = i
        plt.title("opt DMD mode Im %i, $\displaystyle f$ = %4.3f Hz" %(mode,freqs[i]))
        tpc = ax.tripcolor(triang, A[:,i], vmin = -600, vmax = 600, shading='gouraud', cmap = 'jet')
        cbar = f.colorbar(tpc)
        cbar.set_label(r'$\displaystyle\Delta p$ [Pa]')
        plt.savefig('opt_DMD_mode_im%i.png' %mode)
        plt.close()
        
def plotRe(X ,Y ,Z, A, freqs):
    m = A.shape
    triang = matplotlib.tri.Triangulation(X, Y)
    for i in range(0,m[1]):
        f, ax = matplotlib.pyplot.subplots(figsize = (8,8))
        plt.rcParams.update({'font.size':24})
        plt.rc('xtick', labelsize=24) 
        plt.rc('ytick', labelsize=24) 
        ax.set_aspect('equal')
        ax.tricontourf(X, Y, Z, 24)
        ax.set_xlabel(r'$\displaystyle x$ [m]')
        ax.set_ylabel(r'$\displaystyle y$ [m]')
        mode = i
        plt.title("DMD mode Re %i, $\displaystyle f$ = %4.3f Hz" %(mode,freqs[i]))
        tpc = ax.tripcolor(triang, A[:,i], vmin = -600, vmax = 600, shading='gouraud', cmap = 'jet')
        cbar = f.colorbar(tpc)
        cbar.set_label(r'$\displaystyle\Delta p$ [Pa]')
        plt.savefig('opt_DMD_mode_re%i.png' %mode)
        plt.close()
        
def plotFFT(X ,Y ,Z, A, freqs):
    cmap = plt.get_cmap('jet')
    new_cmap = truncate_colormap(cmap, 0.5, 1)
    m = A.shape
    triang = matplotlib.tri.Triangulation(X, Y)
    for i in range(0,m[1]):
        f, ax = matplotlib.pyplot.subplots(figsize = (8,8))
        plt.rcParams.update({'font.size':24})
        plt.rc('xtick', labelsize=24) 
        plt.rc('ytick', labelsize=24) 
        ax.set_aspect('equal')
        ax.tricontourf(X, Y, Z, 24)
        ax.set_xlabel(r'$\displaystyle x$ [m]')
        ax.set_ylabel(r'$\displaystyle y$ [m]')
        mode = i
        plt.title("FFT mode amplitude %i, $\displaystyle f$ = %4.3f Hz" %(mode,freqs[i]))
        tpc = ax.tripcolor(triang, A[:,i], vmin = 0, vmax = 600, shading='gouraud', cmap = new_cmap)
        cbar = f.colorbar(tpc)
        cbar.set_label(r'$\displaystyle\Delta p$ [Pa]')
        plt.savefig('FFT_mode_amplitude%i.png' %mode)
        plt.close()
        
def plotFFT_alex(X ,Y ,Z, A, freqs):
    m = A.shape
    triang = matplotlib.tri.Triangulation(X, Y)
    for i in range(0,m[1]):
        f, ax = matplotlib.pyplot.subplots(figsize = (8,8))
        plt.rcParams.update({'font.size':24})
        plt.rc('xtick', labelsize=24) 
        plt.rc('ytick', labelsize=24) 
        ax.set_aspect('equal')
        ax.tricontourf(X, Y, Z, 24)
        ax.set_xlabel(r'$\displaystyle x$ [m]')
        ax.set_ylabel(r'$\displaystyle y$ [m]')
        mode = i
        plt.title("FFT EO %i, $\displaystyle f$ = %4.3f Hz" %(mode,freqs[i]))
        tpc = ax.tripcolor(triang, A[:,i], vmin = 0, vmax = 400, shading='gouraud', cmap = 'jet')
        cbar = f.colorbar(tpc)
        cbar.set_label(r'$\displaystyle\Delta p$ [Pa]')
        plt.savefig('FFT_mode_EO%i.png' %mode)
        plt.close()
