
import numpy as np
import pandas as pd
import functions_dmd as dfc
from optdmd import DMDOptOperator,OptDMD

def opt(snapshots,j):#j zeigt an, dass der Snapshot für die erste Umdrehung berechnet wird

    dt = 60/(4000*256) # time step of signal, 4000 rev/min with 256 steps


    X, Y = dfc.timeShift(snapshots) # create X(k+1) = Atilde * X(k), where X(k+1) is X2 and X(k) is X

    s = np.shape(snapshots)[0]#number of rows in the snapshotbase


    optA = DMDOptOperator(svd_rank=0, factorization="evd")
    opt=OptDMD(factorization="evd", svd_rank=0, tlsq_rank=0, opt=True)
    opt.fit(X)#Wenn Sie opt verwenden, müssen fit() zuerst passen.
    U,S,V = optA._compute_operator(X,Y)
    S= np.diag(S)#S is a 1d array. for further calculations we need a 2d array with S on the main diagonal
    Atilde=optA.as_numpy_array

    np.savetxt(r"\\nas.tu-clausthal.de\win-home$\zj19\Desktop\optdmd\opt_atilde" +"%d.csv" %j, Atilde)

    #eigenvalues and eigenvectors of Atilde
    lamb= optA.eigenvalues
    W=optA.eigenvectors

    # opt DMD modes
    Phi = dfc.dmdModes(V,S,W,Y)

    # Cut negative half plane of DMD spectrum and modes
    Phi, lamb = dfc.cutModes(Phi, lamb)

    np.savetxt(r"\\nas.tu-clausthal.de\win-home$\zj19\Desktop\optdmd\opt_eigenvalues_real "+ "%d.csv"%j, np.real(lamb))
    np.savetxt(r"\\nas.tu-clausthal.de\win-home$\zj19\Desktop\optdmd\opt_eigenvalues_imaginary"+"%d.csv"%j, np.imag(lamb))

    np.savetxt(r"\\nas.tu-clausthal.de\win-home$\zj19\Desktop\optdmd\opt_phi_real" +"%d.csv"%j ,np.real(Phi))
    np.savetxt(r"\\nas.tu-clausthal.de\win-home$\zj19\Desktop\optdmd\opt_phi_imaginary"+"%d.csv"%j, np.imag(Phi))
    Phi_abs = dfc.scaleModes(Phi,snapshots)#缩放方法b=Phix1

    np.savetxt(r"\\nas.tu-clausthal.de\win-home$\zj19\Desktop\optdmd\opt_phi_abs" +"%d.csv"%j, np.real(Phi_abs))

    #DMD Spectra
    DMDfreqs, DMDpower = dfc.dmdSpectrum(lamb, dt, Phi, snapshots, s)

    rowsPhi_abs = len(Phi_abs)
    colsPhi_abs = len(Phi_abs[0])
    Phi_mean = np.zeros(colsPhi_abs)

    for i in range (colsPhi_abs):
        Phi_mean[i] = sum(Phi_abs[:,i])/rowsPhi_abs

    np.savetxt(r"\\nas.tu-clausthal.de\win-home$\zj19\Desktop\optdmd\optDMDfreqs_real" +"%d.csv" %j, np.real(DMDfreqs))
    np.savetxt(r"\\nas.tu-clausthal.de\win-home$\zj19\Desktop\optdmd\optDMDfreqs_imaginary" +"%d.csv" %j, np.imag(DMDfreqs))
    np.savetxt(r"\\nas.tu-clausthal.de\win-home$\zj19\Desktop\optdmd\optDMDpower" +"%d.csv" %j, DMDpower)
    np.savetxt(r"\\nas.tu-clausthal.de\win-home$\zj19\Desktop\optdmd\optDMDpower_Phi" +"%d.csv" %j, Phi_mean)


snapshots = pd.read_csv(r"\\nas.tu-clausthal.de\win-home$\zj19\Desktop\MA\snapshotbase_Shroud_2.csv", delimiter = " ", header = None).values
opt(snapshots,2) 
