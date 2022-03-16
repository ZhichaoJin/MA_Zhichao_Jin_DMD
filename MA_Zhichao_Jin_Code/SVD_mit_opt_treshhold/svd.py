#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 16:27:02 2019

@author: ksl12
"""

import pandas as pd
import numpy as np

A = pd.read_csv(r"\\nas.tu-clausthal.de\win-home$\zj19\Desktop\MA\snapshotbase_Shroud_2.csv", delimiter = " ", header = None).values

m = np.size(A, 1) #columns of A

X = A[:,:m-1]

U, S, V= np.linalg.svd(X, full_matrices=False, compute_uv=True)

#Berechnung des optimalen Treshhold
omega = lambda x: 0.56 * x**3 - 0.95 * x**2 + 1.82 * x + 1.43
beta = np.divide(*sorted(X.shape))
tau = np.median(S) * omega(beta)
rank = np.sum(S > tau)
print(rank)


np.savetxt("S.csv", S)
np.savetxt("U.csv", U)
np.savetxt("V.csv", V)

# the number of singular values will be the number of snapshots minus 1, since
# in order to perform the DMD later, the snapshotbase had to be split into
# X and X2 to create a reference to time