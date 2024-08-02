import os
import numpy as np
import time

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20
plt.rcParams['axes.linewidth'] = 3

import sys
sys.path.append("../../../")
import dam as Dam
#------------------------------------------------------------------------------
#To fit EFS
bs_dict_opt1={'twojmax':'8', #(8,55),(10,91),(12,140),(14,204),(16,285)
         'nbspa':55, #Total number of bispectrum coefficients per atom (depends on twojmax)
         'rcutfac':4.0,
         'rfac0': 0.9539387603447764,
         'nat':7,
         
         'e1': 'C3a',
         'r1': 0.46137337377768084,
         'w1': 1.0469585267425072,
         'm1': '12.011',

         'e2': 'C3b', 
         'r2': 0.44489322132537323,
         'w2': 0.8938486305346591,
         'm2': '12.011',

         'e3': 'C4a',
         'r3': 0.47332709267472434, 
         'w3': 0.9770358209723538, 
         'm3': '12.011', 
         
         'e4': 'H1a',
         'r4': 0.27719310996913965,
         'w4': 0.7812429580883443,
         'm4': '1.008',

         'e5': 'H1b',
         'r5': 0.28175161540698707,
         'w5': 0.7322293401201861,
         'm5': '1.008',
         
         'e6': 'N3a',
         'r6': 0.4524047838214516,
         'w6': 1.1843027932950918 ,
         'm6': '14.007',
         
         'e7': 'Zn4a',
         'r7': 0.4169879943214552,
         'w7': 0.2908421990926551,
         'm7': '65.409',
         'Alpha':0.01, # Alpha for ridge regression 
         
         'Ew': 1249.934862564297, # Weight of energy equations
         'Sw': 3833.7521420902817,  # Weight of stress equations
         }
#------------------------------------------------------------------------------
bs_dict=bs_dict_opt1.copy();

#Testing------------------------------------------------------
ftt='TestA_at.xyz';
Ntt=int(os.popen('cat '+ftt+' | grep snap | wc -l').read().strip());
print(Ntt,'Configuration in training set')
t=Dam.aSNAP.SnapTest(ftt, Ntt, 'EFS', bs_dict)
t.test_fit()
del t
#-------------------------------------------------------------
