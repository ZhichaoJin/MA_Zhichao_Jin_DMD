#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:57:19 2020

@author: root
"""
# The functions in this file are parts of the DMD algorithm.
# If you just want a standard DMD, or you don't know what you want yet,
# you may use standardDMD() from DMD.py

import numpy as np
from numpy.core.fromnumeric import mean
from scipy import linalg

def timeShift(A): # this is a help function you can use if you need to pass X or X2 more than once
    m = np.size(A, 1) #columns of A
    X = A[:,:m-1] # snapshotbase without last timestep
    X2 = A[:,1:m] # snapshotbase shifted into the future by one timestep
    return X, X2

def SVD(X, r = None):
    
    # Singular Value Decomposition
    U, S, V = linalg.svd(X, full_matrices=False, compute_uv=True)
    S_diag = np.diag(S) #S is a 1d array. for further calculations we need a 2d array with S on the main diagonal
    
    if r != None: #truncation
        U_r, S_r, V_r = truncate(U,S_diag,V,r)
        return U_r, S_r, V_r
    else: return U, S_diag, V
    
    # if you want to check at which point to truncate, you can first do an svd
    # without r and just let it return S_diag. Once you have checked from that
    # do an svd again with truncation. However, no need to do that, if you already
    # did an svd before. So instead, you may also save the full u,s,v from the full
    # svd and then just use truncate()

def truncate(U,S_diag,V,r):
    
    U_r = U[:,:r]
    S_r = S_diag[:r,:r]
    V_r = V[:r,:]
    
    # add a check, if S is diagonal, so it also works with not diagonalized Matrix

    return U_r, S_r, V_r

def podReconstruct(X, U_r):
    #Calculate reconstruction of snapshotbase from POD Modes

    X_pod = np.dot(np.dot(U_r,np.transpose(U_r)),X)
    
	# insert calculation of norm in order to know the quality of reconstruction
    
    return X_pod

def calcAtilde(U_r, S_r, V_r, X2):
    
    #Atilde is the low-dimensional linear model of the dynamical system on POD coordinates
    Atilde = np.dot(np.dot(np.dot(np.transpose(U_r),X2),np.transpose(V_r)),np.linalg.inv(S_r))

    return Atilde

def eigenDecomposition(Matrix):
	
    D, W = np.linalg.eig(Matrix) #eigenvalues and eigenvectors of Matrix

    return D, W

def dmdModes(V_r, S_r, W, X2):
    
    Phi = np.dot(np.dot(np.dot(X2,np.transpose(V_r)),np.linalg.inv(S_r)),W)
	
    return Phi

def cutModes(Phi,eig):
    #Delete the negative half plane of the DMD spectrum, since it is redundant
    
    i = 0
    
    while i < np.size(eig):
        if i > 0 and eig.imag[i-1] < 0:
            i = i-1
        if eig.imag[i] < 0:
            eig = np.delete(eig,i)
            Phi = np.delete(Phi,i,1)
        i = i+1
            
    return Phi, eig

def scaleModes(Phi, A):
    
    x1 = A[:,0]
    b = np.absolute(np.linalg.lstsq(Phi,x1)[0])
    m = np.size(Phi, 1) #columns of A
    n = np.size(Phi, 0) #rows of A
    Phi = np.absolute(Phi)
    Phi_scaled = np.zeros((n,m))
    for i in range (m):
        Phi_scaled[:,i] = Phi[:,i]*b[i]*2
        
    return Phi_scaled



def predictiveReconstruction(A, D, dt, Phi, r):
    m = np.size(A, 1) #columns of A
    n = np.size(A, 0) #rows of A
    #DMD Spectra
    lamb = D #discrete-time eigenvalues
    omeg = np.dot(np.log(lamb),1/dt) #continuous-time eigenvalues

    t = np.zeros(m)
    t[0] = 0

    for k in range (1,m):
        t[k] = t[k-1]+dt

	#predictive reconstruction:
	# b: DMD Mode amplitudes

    x1 = A[:,0]
    b = np.linalg.lstsq(Phi,x1)[0] #least squares solution for Phi*b=x1. return[0] gives the solution

    X_dmd = np.zeros((n,m), dtype=complex)
    time_dynamics = np.zeros((r,m), dtype=complex)
    for j in range(m):
        time_dynamics[:,j] = b*np.exp(np.dot(omeg,t[j]))
    X_dmd = np.dot(Phi,time_dynamics)
    return X_dmd



# def dmdSpectrum(lamb, dt, Phi, A, s,W,U):
#     x1 = A[:,0]
#     tild_x1=np.dot(np.transpose(U),x1)
#     b1 = np.linalg.lstsq(Phi,x1)[0] #least squares solution for Phi*b=x1. return[0] gives the solution
#     b2=np.dot(np.linalg.inv((W*lamb)),tild_x1)
#     b3=np.dot(np.linalg.pinv(Phi),x1)
#     #Power Spectrum
#     DMDfreqs = np.divide(np.log(lamb),(dt*2*np.pi))
#     DMDpower = np.divide(np.abs(b1)*2,np.sqrt(s))
#     DMDpower2 = np.divide(np.abs(b2)*2,np.sqrt(s))
#     DMDpower3 = np.divide(np.abs(b3)*2,np.sqrt(s))
#     return DMDfreqs, DMDpower,DMDpower2,DMDpower3


    
def dmdSpectrum(lamb, dt, Phi, A, s):
    x1 = A[:,0]
    #x1 = np.median(A,axis=1)
    b1 = np.linalg.lstsq(Phi,x1)[0] #least squares solution for Phi*b=x1. return[0] gives the solution
    #Power Spectrum
    DMDfreqs = np.divide(np.log(lamb),(dt*2*np.pi))
    DMDpower = np.divide(np.abs(b1)*2,np.sqrt(s))

    return DMDfreqs, DMDpower
