import os
import numpy as np
import time

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20
plt.rcParams['axes.linewidth'] = 3

import sys
#sys.path.append("/home/asharma/apycodes/")
#from aSNAP_v35 import *

sys.path.append("../../../")
import dam as Dam

#clear(True)
#------------------------------------------------------------------------------
#To fit EFS
bs_dict_opt1={'twojmax':'8', #(8,55),(10,91),(12,140),(14,204),(16,285)
         'nbspa':55, #Total number of bispectrum coefficients per atom (depends on twojmax)
         'rcutfac':4.0,
         'rfac0': 0.999, #0.8859556496855889,
         'nat':7,
         
         'e1': 'C3a',
         'r1': 0.4,
         'w1': 1.0,
         'm1': '12.011',

         'e2': 'C3b',
         'r2': 0.4,
         'w2': 1.0,
         'm2': '12.011',

         'e3': 'C4a',
         'r3': 0.4,
         'w3': 1.0,
         'm3': '12.011',
         
         'e4': 'H1a',
         'r4': 0.3,
         'w4': 1.0,
         'm4': '1.008',

         'e5': 'H1b',
         'r5': 0.3,
         'w5': 1.0,
         'm5': '1.008',

         'e6': 'N3a',
         'r6': 0.4,
         'w6': 1.0,
         'm6': '14.007',
         
         'e7': 'Zn4a',
         'r7': 0.4,
         'w7': 1.0,
         'm7': '65.409',
         'Alpha':0.01, # Alpha for ridge regression (E:0.0),(EFS:1.0)
         
         'Ew': 100.0, #592.833268416893, # Weight of energy equations
         'Sw': 300.0,  # Weight of stress equations
         }
#------------------------------------------------------------------------------
bs_dict = bs_dict_opt1.copy();

ft='Train_D672_at.xyz';
Nt=int(os.popen('cat '+ft+' | grep snap | wc -l').read().strip());
print(Nt,'Configuration in training set')
s = Dam.aSNAP.SnapEQ(ft,Nt,'EFS',bs_dict,0)
s.etype='rmse' 
s.opt_para('DE')
del s
#-------------------------------------------------------------




