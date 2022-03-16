#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 12:01:13 2019

@author: ksl12
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import decimal

re = pd.read_csv(r"\\nas.tu-clausthal.de\win-home$\\zj19\Desktop\\MA\\eigenvalues_real.csv").values
im = pd.read_csv(r"\\nas.tu-clausthal.de\win-home$\\zj19\Desktop\\MA\\eigenvalues_imaginary.csv").values

fig, ax = plt.subplots()
plt.xlabel("Re")
plt.ylabel("Im")
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
#ax.scatter(re, im)

stabil = 0
instabil = 0
grenzstabil = 0

rows_wert = len(re)
cols_wert = len(re[0])
wert = np.zeros(rows_wert)
print(rows_wert)
print(wert.shape)
x1=0
x2=0
x3=0
for i in range(rows_wert):
  wert[i] = abs(complex(re[i],im[i]))
  decimal.getcontext().rounding = "ROUND_HALF_UP"
  wert[i] = decimal.Decimal(wert[i]).quantize(decimal.Decimal("0.1")) #Gerundet, um numerische Fehler des Computers auszuschließen
  if wert[i] < 1:
     x1=ax.scatter(re[i], im[i],label= 'stabil',color = "green")
     stabil = stabil + 1
  if wert[i] > 1:
     x2=ax.scatter(re[i], im[i],label= 'instabil',color = "red")
     instabil = instabil + 1
  if wert[i] == 1:
     x3=ax.scatter(re[i], im[i],label= 'grenzstabil',color = "orange")
     grenzstabil = grenzstabil + 1
print(stabil,instabil,grenzstabil)
sum = stabil + instabil + grenzstabil
aperiodisch = (stabil + instabil) / sum
print(wert)
print(" aperiodisch Anteil :{0:.0%}" .format(aperiodisch))

circle = plt.Circle((0, 0), 1, fill=False)

ax.add_artist(circle)

if x1==0:
   plt.legend(handles=[x3,x2], labels=['grenzstabil','instabil'],loc = 'lower right')
if x2==0:
   plt.legend(handles=[x1,x3], labels=['stabil','grenzstabil'],loc = 'lower right')
if x3==0:
   plt.legend(handles=[x1,x2], labels=['stabil','instabil'],loc = 'lower right')

plt.title("dmd Eigenwertesverteilung aperiodisch Anteil :{0:.0%}" .format(aperiodisch))
plt.savefig(r"\\nas.tu-clausthal.de\win-home$\\zj19\Desktop\\MA\\eigenvaluesDMD_r107_U2.png")
