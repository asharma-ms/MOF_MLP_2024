import os
import numpy as np
import time

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20
plt.rcParams['axes.linewidth'] = 3

import sys

sys.path.append("../../../../../../") # Change this path according to location of Dam package
import dam as Dam

#------------------------------------------------------------------------------
#To fit EFS
bs_dict_opt1={'twojmax':'8', #(8,55),(10,91),(12,140),(14,204),(16,285)
         'nbspa':55, #Total number of bispectrum coefficients per atom (depends on twojmax)
         'rcutfac':4.0,
         'rfac0': 0.999, 
         'nat':4,
         
         'e1': 'C',
         'r1': 0.4,
         'w1': 1.0,
         'm1': '12.011',
         
         'e2': 'H',
         'r2': 0.3,
         'w2': 1.0,
         'm2': '1.008',

         'e3': 'O',
         'r3': 0.4,
         'w3': 1.0,
         'm3': '15.999',
         
         'e4': 'Zn',
         'r4': 0.4,
         'w4': 1.0,
         'm4': '65.409',
         'Alpha':0.01, # Alpha for ridge regression (E:0.0),(EFS:1.0)
         
         'Ew': 100.0,  # Weight of energy equations
         'Sw': 300.0,  # Weight of stress equations
         }
#------------------------------------------------------------------------------
bs_dict=bs_dict_opt1.copy();

#Just fitting-------------------------------------------------
ft='../MLPData/TS.xyz'
Nt=int(os.popen('cat '+ft+' | grep snap | wc -l').read().strip());
#Nt = 490
print(Nt,'Configuration in training set')
s=Dam.aSNAP.SnapEQ(ft,Nt,'EFS',bs_dict,0)
s.we_fac=bs_dict['Ew']
s.ws_fac=bs_dict['Sw']

t1 = time.perf_counter()
s.get_BSEQ()
t2 = time.perf_counter()
print(f'Finished in {round(t2-t1, 3)} second(s)')
s.fit_bsPara()
#s.fit_bsPara_varyAlpha()
##er,fr,sr=s.get_rmse()
s.write_bsPara()
s.plot_fit()
##print(er,fr,sr)
del s
#-------------------------------------------------------------




