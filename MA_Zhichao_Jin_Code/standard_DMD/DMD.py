# -*- coding: utf-8 -*-

# before using this script, do SVD

import pandas as pd
import numpy as np
import functions_dmd as dfc

r = 107 # enter the chosen truncation threshold from the singular values plot
dt = 60/(4000*256) # time step of signal, 4000 rev/min with 256 steps

A = pd.read_csv(r"\\nas.tu-clausthal.de\win-home$\\zj19\Desktop\\MA\\snapshotbase_shroud_2.csv", delimiter = " ", header = None).values
s = np.size(A, 0) #number of rows in the snapshotbase
X, X2 = dfc.timeShift(A) # create X(k+1) = Atilde * X(k), where X(k+1) is X2 and X(k) is X

# SVD needs to be done previously, because only then we know how to choose the truncation threshold
# U, V and S are the results of the SVD
U = pd.read_csv(r"\\nas.tu-clausthal.de\win-home$\zj19\Desktop\optdmd\U.csv", delimiter = " ", header = None).values
V = pd.read_csv(r"\\nas.tu-clausthal.de\win-home$\zj19\Desktop\optdmd\V.csv", delimiter = " ", header = None).values
S = pd.read_csv(r"\\nas.tu-clausthal.de\win-home$\zj19\Desktop\optdmd\S.csv", delimiter = " ", header = None).values

S_flat = np.ndarray.flatten(S)

S_diag = np.diag(S_flat) #S is a 1d array. for further calculations we need a 2d array with S on the main diagonal

U = U[:,:r]
S_diag = S_diag[:r,:r]
V = V[:r,:]
np.savetxt(r"\\nas.tu-clausthal.de\win-home$\\zj19\Desktop\\MA\\U_r.csv", U) # U are the POD modes, they need to be saved for plotting
print(U.shape,S_diag.shape,V.shape,X2.shape)
#Atilde is the low-dimensional linear model of the dynamical system on POD coordinates
Atilde = dfc.calcAtilde(U,S_diag,V,X2)
np.savetxt(r"\\nas.tu-clausthal.de\win-home$\\zj19\Desktop\\MA\\atilde.csv", Atilde)

#eigenvalues and eigenvectors of Atilde
lamb, W = dfc.eigenDecomposition(Atilde)
np.savetxt(r"\\nas.tu-clausthal.de\win-home$\\zj19\Desktop\\MA\\eigenvalues.csv", lamb)
np.savetxt(r"\\nas.tu-clausthal.de\win-home$\\zj19\Desktop\\MA\\eigenvectors.csv", W)
# DMD modes
Phi = dfc.dmdModes(V,S_diag,W,X2)

# Cut negative half plane of DMD spectrum and modes
Phi, lamb = dfc.cutModes(Phi, lamb)

np.savetxt(r"\\nas.tu-clausthal.de\win-home$\\zj19\Desktop\\MA\\eigenvalues_real.csv", np.real(lamb))
np.savetxt(r"\\nas.tu-clausthal.de\win-home$\\zj19\Desktop\\MA\\eigenvalues_imaginary.csv", np.imag(lamb))
np.savetxt(r"\\nas.tu-clausthal.de\win-home$\\zj19\Desktop\\MA\\phi_real.csv", np.real(Phi))
np.savetxt(r"\\nas.tu-clausthal.de\win-home$\\zj19\Desktop\\MA\\phi_imaginary.csv", np.imag(Phi))
Phi_abs = dfc.scaleModes(Phi,A)
np.savetxt(r"\\nas.tu-clausthal.de\win-home$\\zj19\Desktop\\MA\\phi_abs.csv", np.real(Phi_abs))



# rowsPhi_abs = len(Phi_abs)
# colsPhi_abs = len(Phi_abs[0])
# Phi_mean = np.zeros(colsPhi_abs)

# for i in range (colsPhi_abs):
#     Phi_mean[i] = sum(Phi_abs[:,i])/rowsPhi_abs #每列求和除以行数，或者说通过列求和方式取平均值同mean(Phi_abs,axis=0) 1是行，0是列

Phi_mean=np.mean(Phi_abs,axis=0)

#DMD Spectra
DMDfreqs, DMDpower= dfc.dmdSpectrum(lamb, dt, Phi, A, s)       
np.savetxt(r"\\nas.tu-clausthal.de\win-home$\\zj19\Desktop\\MA\\DMDfreqs_real2.csv", np.real(DMDfreqs))
np.savetxt(r"\\nas.tu-clausthal.de\win-home$\\zj19\Desktop\\MA\\DMDfreqs_imaginary2.csv", np.imag(DMDfreqs))
np.savetxt(r"\\nas.tu-clausthal.de\win-home$\\zj19\Desktop\\MA\\DMDpower2.csv", DMDpower)
np.savetxt(r"\\nas.tu-clausthal.de\win-home$\\zj19\Desktop\\MA\\DMDpower_mean2.csv", Phi_mean)


#Skalierungsmethode
# DMDfreqs, DMDpower, DMDpower2,DMDpower3= dfc.dmdSpectrum(lamb, dt, Phi, A, s,W,U)       
# np.savetxt(r"\\nas.tu-clausthal.de\win-home$\\zj19\Desktop\\MA\\DMDfreqs_real.csv", np.real(DMDfreqs))
# np.savetxt(r"\\nas.tu-clausthal.de\win-home$\\zj19\Desktop\\MA\\DMDfreqs_imaginary1.csv", np.imag(DMDfreqs))
# np.savetxt(r"\\nas.tu-clausthal.de\win-home$\\zj19\Desktop\\MA\\DMDpower.csv", DMDpower)
# np.savetxt(r"\\nas.tu-clausthal.de\win-home$\\zj19\Desktop\\MA\\DMDpower2.csv", DMDpower2)
# np.savetxt(r"\\nas.tu-clausthal.de\win-home$\\zj19\Desktop\\MA\\DMDpower3.csv", DMDpower3)
# np.savetxt(r"\\nas.tu-clausthal.de\win-home$\\zj19\Desktop\\MA\\DMDpower_mean1.csv", Phi_mean)

##Calculate reconstruction of snapshotbase from POD Modes (needs a lot of RAM!!)
#X_pod = np.dot(np.dot(U,np.transpose(U)),X)
#X_pod_var = X-X_pod #to plot differences between original and reconstruct
#np.savetxt("X_pod_var.csv", X_pod_var)

##predictive reconstruction from DMD modes
##called predictive, because it is possible to reconstruct into the future
##in order for that to work, fc.predictiveReconstruction needs to be modified though
#X_reconstruct = dfc.predictiveReconstruction(A, lamb, dt, Phi, r)
#np.savetxt("X_reconstruct.csv", X_reconstruct)
