
import time
import itertools
import matplotlib.cm as cm 
from _window import WindowDMD
import functions_dmd as dfc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from plot_eigenvalues import plot_eigen


dt = 60/(4000*256) # time step of signal, 4000 rev/min with 256 steps

snapshots = pd.read_csv(r"\\nas.tu-clausthal.de\win-home$\zj19\Desktop\MA\snapshotbase_Shroud_mehr.csv", delimiter = " ", header = None).values
X, Y = dfc.timeShift(snapshots) # create X(k+1) = Atilde * X(k), where X(k+1) is X2 and X(k) is X

n, m = len(X[:, 0]), len(X[0, :])
w =256 #Breite des Fensters
tspan = np.linspace(0, 60/256*m, m) #Startzeit 0 bis Endzeit
t = tspan[1:] #Jeder Zeitpunkt


wdmd = WindowDMD(n, w)
wdmd.initialize(X[:, :w], Y[:, :w])
start = time.time()
for k in range(w, w+10): #+10 bedeutet nur 10 Schritte
    eigenvalues,eigenvectors = wdmd.computemodes() 
    f=np.abs(np.divide(np.log(np.imag(eigenvalues)),(dt*2*np.pi))) # Berechnung der Frequenzwerte
    #Mapping in Echtzeit
    plt.figure(num=1,figsize=(16, 12))
    plt.subplot(211)
    plt.ion()
    #plt.clf()
    color_cycle= itertools.cycle(["orange","pink","blue","brown","red","grey","yellow","green"]) # Geordnete farbige Arrays von Frequenzpunkten
    for j in range(wdmd.rank):
        plt.plot(t[k], f[j],".",color=next(color_cycle)) #Jeder Schritt stellt den Frequenzwert des aktuellen Zeitknotens dar und färbt ihn ein
    plt.xlabel('t [s]')
    plt.ylabel('f [Hz]')
    plt.ylim(0,2000)
    plt.draw()
    plt.pause(0.01)
    #Echtzeit-Abbildung der Eigenwertverteilung
    plt.subplot(212)
    ax2=plt.subplot(212)
    plot_eigen(np.real(eigenvalues),np.imag(eigenvalues),ax2)
    plt.draw()
    plt.savefig('W_DMD_%i.png' %k)# Speichern der Frequenzerfassungskarte und der Eigenwertverteilung für jeden Schritt
    wdmd.update(X[:, k], Y[:, k])

    
end = time.time()
print("Window DMD, w=200, time = " + str(end-start) + " secs")
