import numpy as np
import pandas as pd
import time

#read one timestep to get the dimensions
A = pd.read_csv("L:/Code/csv/Hub/Hub_11856.csv", skiprows = 6, names=["Node Number","X [ m ]","Y [ m ]","Z [ m ]","Absolute Pressure [ Pa ]"])

n = A.shape

print("Number of rows in snapshotbase: ", n[0])

t_ini = 11856 #number of first timestep
t_fin = 12111 #number of last timestep
step = 5 #every 5th timestep from the simulation was saved
t = int((t_fin-t_ini)/step+1) #number of all timesteps
B = np.zeros((n[0],t)) #matrix for the snapshotbase
CalCount = 0
CalMatrix = 0
print("total %d" %t)
#write the data from every timestep file into the snapshot matirx B
for y in range(0,t):
    count = t_ini+step*y #here you need to use the correct timestep numbers again, for the filenames
    A = pd.read_csv("L:/Code/csv/Hub/Hub_%i.csv" % count, skiprows = 6, names=["Node Number","X [ m ]","Y [ m ]","Z [ m ]","Absolute Pressure [ Pa ]"])
    U = A["Absolute Pressure [ Pa ]"].values
    for x in range(0,n[0]):
            B[x,y] = U[x]
            CalMatrix = CalMatrix + 1
            CalMatrixPrint = ((CalMatrix/154697)*100)
            print("%.15f%% " % CalMatrixPrint)
            
    CalCount = CalCount + 1
    
    print("总共完成%d %%" % int((CalCount/t)*100))

np.savetxt("snapshotbase.csv", B)
print("done")
