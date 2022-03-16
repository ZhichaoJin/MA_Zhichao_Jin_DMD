#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 12:01:13 2019

@author: ksl12
"""
import numpy as np
import matplotlib.pyplot as plt

import decimal



def plot_eigen(re,im,ax):

   
   plt.xlabel("Re")
   plt.ylabel("Im")
   plt.xlim(-1.1, 1.1)
   plt.ylim(-1.1, 1.1)
   stabil = 0
   instabil = 0
   grenzstabil = 0

   rows_wert = len(re)
   if type(re[0]) != np.array:
      cols_wert = 1
   else:
      cols_wert = len(re[0])
   print(re.shape)
   wert = np.empty(rows_wert)

   print(wert.shape)

   for i in range(0,rows_wert):

      wert[i] =np.asarray(abs(complex(re[i],im[i])))
      decimal.getcontext().rounding = "ROUND_HALF_UP"
      wert[i] = decimal.Decimal(wert[i]).quantize(decimal.Decimal("0.1")) #Gerundet, um numerische Fehler des Computers auszuschließen
      if wert[i] < 1:
         plt.scatter(re[i], im[i],label= 'stabil',color = "green")
         stabil = stabil + 1
      if wert[i] > 1:
         plt.scatter(re[i], im[i],label= 'instabil',color = "red")
         instabil = instabil + 1
      if wert[i] == 1:
         plt.scatter(re[i], im[i],label= 'grenzstabil',color = "orange")
         grenzstabil = grenzstabil + 1
   print(stabil,instabil,grenzstabil)
   sum = stabil + instabil + grenzstabil
   aperiodisch = (stabil + instabil) / sum
   print(wert)
   print(" aperiodisch Anteil :{0:.0%}" .format(aperiodisch))

   circle = plt.Circle((0, 0), 1, fill=False)

   ax.add_artist(circle)
   plt.title("Wdmd Eigenwertesverteilung aperiodisch Anteil :{0:.0%}" .format(aperiodisch))




